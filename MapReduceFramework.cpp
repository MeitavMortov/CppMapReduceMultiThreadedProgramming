#include "MapReduceFramework.h"
#include "MapReduceClient.h"
#include <atomic>
#include <algorithm>
#include <pthread.h>
#include <cstdlib>
#include <iostream>

#define TOTAL_BIT 33
#define COUNTER_BIT 2
#define PERCENTAGE_FACTOR 100
#define MUTEX_DESTROY_ERROR "system error: error on pthread_mutex_destroy"
#define COND_DESTROY_ERROR "system error: error on pthread_cond_destroy"
#define MUTEX_LOCK_ERROR "system error: error on pthread_mutex_lock"
#define COND_WAIT_ERROR "system error: error on pthread_cond_wait"
#define COND_BROADCAST_ERROR "system error: error on pthread_cond_broadcast"
#define MUTEX_UNLOCK_ERROR "system error: error on pthread_mutex_unlock"
#define MUTEX_INIT_ERROR "system error: error on pthread_mutex_init"
#define CREATE_THREAD_ERROR "system error: error on pthread_create"
#define JOIN_THREAD_ERROR "system error: error on pthread_join"

void system_error_handler(const std::string& error_type){
    std::cerr << error_type << std::endl;
    exit(1);
}

/**
 * Implementation of a barrier using mutexes
 */
class Barrier{
private:
    pthread_mutex_t mutex;
    pthread_cond_t cv;
    int count;
    int numThreads;
public:
    Barrier(int numThreads)
    : mutex(PTHREAD_MUTEX_INITIALIZER)
    , cv(PTHREAD_COND_INITIALIZER)
    , count(0)
    , numThreads(numThreads)
    { }


    ~Barrier()
    {
        if (pthread_mutex_destroy(&mutex) != 0) {
            system_error_handler(MUTEX_DESTROY_ERROR);
        }
        if (pthread_cond_destroy(&cv) != 0){
            system_error_handler(COND_DESTROY_ERROR);
        }
    }


    void barrier()
    {
        if (pthread_mutex_lock(&mutex) != 0){
            system_error_handler(MUTEX_LOCK_ERROR);
        }
        if (++count < numThreads) {
            if (pthread_cond_wait(&cv, &mutex) != 0){
                system_error_handler(COND_WAIT_ERROR);
            }
        }
        else {
            count = 0;
            if (pthread_cond_broadcast(&cv) != 0) {
                system_error_handler(COND_BROADCAST_ERROR);
            }
        }
        if (pthread_mutex_unlock(&mutex) != 0) {
            system_error_handler(MUTEX_UNLOCK_ERROR);
        }
    }
};

/**
 * This class includes all the parameters which are relevant to the job
 */
class JobContext{
    public:
        IntermediateVec *mid_output; // output of the first phase (map)
        std::vector<IntermediateVec*> shuffled_vector; // output of the second phase (shuffle)
        std::atomic<int> map_partition_cnt{0}; // counter for the partition of the pairs between threads (map phase)
        std::atomic<uint64_t> atomic_state{0}; // represents the current state, divided into stage | count | total
        Barrier* barrier; // a barrier (see Barrier class above)
        pthread_mutex_t lock_shuffled; // mutex that protects  the shuffled vector
        pthread_mutex_t lock_output; // mutex that protects the output vector
        int multiThreadLevel; // number of threads to be used for running the algorithm
        pthread_t *threads; // array of pthreads used during the algorithm
        int is_waitForJob_called; // boolean - has the function waitForJob been called
        uint64_t shuffle_total; // number of pairs in the shuffled vector

        /**
        * JobContext constructor - initializes the class
        * @param: multiThreadLevel the number of worker threads to be used for running the algorithm.
        */
        JobContext(int multiThreadLevel){
            this->multiThreadLevel = multiThreadLevel;
            mid_output = new IntermediateVec[multiThreadLevel];
            barrier = new Barrier(multiThreadLevel);
            threads = new pthread_t[multiThreadLevel];
            is_waitForJob_called = 0;
            shuffle_total=0;
            if (pthread_mutex_init(&lock_shuffled, nullptr) != 0){
                system_error_handler(MUTEX_INIT_ERROR);
            }
            if (pthread_mutex_init(&lock_output, nullptr)!= 0){
                system_error_handler(MUTEX_INIT_ERROR);
            }
        }

        /**
         * JobContext destructor
         */
        ~JobContext(){
            for (long unsigned int i=0; i<shuffled_vector.size(); i++){
                delete shuffled_vector[i];
            }
            shuffled_vector.clear();
            if (pthread_mutex_destroy(&lock_shuffled) != 0) {
                system_error_handler(MUTEX_DESTROY_ERROR);
            }
            if (pthread_mutex_destroy(&lock_output) != 0) {
                system_error_handler(MUTEX_DESTROY_ERROR);
            }
            delete barrier;
            delete[] mid_output;
            delete[] threads;
        }
};

/**
 * args struct - contains the arguments given to the threads
 */
struct ThreadArgs {
    int thread_num;
    const MapReduceClient& client;
    const InputVec& inputVec;
    OutputVec& outputVec;
    JobContext *job;
};

/**
 * map context struct - contains the context needed for emit2
 */
struct MapContext{
    IntermediateVec *mid_vec;
};

/**
 * reduce context struct - contains the context needed for emit3
 */
struct ReduceContext{
    std::atomic<uint64_t>* counter;
    long unsigned int vec_size;
    pthread_mutex_t *lock_output;
    OutputVec *output_vec;
};

/**
 * This function is called from the client's map function. It saves the output in the current thread's mid-output vector
 * @param key the mid-output key
 * @param value the mid-output value
 * @param context contains the context needed for this function to work
 */
void emit2 (K2* key, V2* value, void* context){
    struct MapContext* map_context = (MapContext*)context;
    IntermediatePair pair(key, value);
    map_context->mid_vec->push_back(pair);
}

/**
 * This function is called from the client's reduce function. It saves the output element in the output vector.
 * @param key the output key
 * @param value the output value
 * @param context contains the context needed for this function to work
 */
void emit3 (K3* key, V3* value, void* context){
    struct ReduceContext* reduce_context = (ReduceContext*)context;
    OutputPair pair(key, value);
    if (pthread_mutex_lock(reduce_context->lock_output) != 0){
        system_error_handler(MUTEX_LOCK_ERROR);
    }
    reduce_context->output_vec->push_back(pair);
    if(pthread_mutex_unlock(reduce_context->lock_output) != 0){
        system_error_handler(MUTEX_UNLOCK_ERROR);
    }
    *(reduce_context->counter) += reduce_context->vec_size << COUNTER_BIT;
}

/**
 * Runs over the last pairs in the mit-output vectors, and finds the pair with the maximum key.
 * Used in the shuffle phase
 * @param job the job context with the parameters which are relevant to the job
 * @return the pair with the maximum key
 */
K2* find_max_key(JobContext& job){
    K2* max_key = nullptr;
    for(int j=0; j < job.multiThreadLevel; j++){
        if(!job.mid_output[j].empty()){
            K2* last_val = job.mid_output[j].back().first;
            if(max_key){
                //check if bigger
                if(*max_key < *last_val){
                    max_key = last_val;
                }
            }
            else{ // max key is null
                max_key = last_val;
            }
        }
    }
    return max_key; // assume there is at least one vector which is not empty
}

/**
 * creates a new sequences of (k2, v2) where in each sequence all keys are identical and all elements with a given key
 * are in a single sequence.
 * @param job the job context with the parameters which are relevant to the job
 */
void shuffle_phase(JobContext& job){
    uint64_t total = 0;
    for (int i=0; i < job.multiThreadLevel; i++){
        total += job.mid_output[i].size();
    }
    uint64_t state_value = (total << TOTAL_BIT) + SHUFFLE_STAGE;
    job.shuffle_total = total;
    job.atomic_state.exchange(state_value);
    uint64_t i = 0;
    while (i < total){
        K2 *max = find_max_key(job);
        IntermediateVec* max_vector = new IntermediateVec();
        for(int j=0; j < job.multiThreadLevel; j++){
            if(!job.mid_output[j].empty()){
                K2 *last_val = job.mid_output[j].back().first;
                while (!((*last_val < *max) || (*max < *last_val))){
                    max_vector->push_back(job.mid_output[j].back());
                    job.mid_output[j].pop_back();
                    i++;
                    job.atomic_state += 1 << COUNTER_BIT;
                    if(!job.mid_output[j].empty()){
                        last_val = job.mid_output[j].back().first;
                    }
                    else{
                        break;
                    }
                }
            }
        }
        job.shuffled_vector.push_back(max_vector);
    }
}

/**
 * Runs the client's reduce function on vectors and saves the output in outputVec
 * @param client the client
 * @param job the job context with the parameters which are relevant to the job
 * @param outputVec the vector where the output should be saved
 */
void reduce_phase(const MapReduceClient& client, JobContext& job, OutputVec *outputVec){
        if (pthread_mutex_lock(&job.lock_shuffled) != 0){
            system_error_handler(MUTEX_LOCK_ERROR);
        }
        while (!job.shuffled_vector.empty()){
            auto *current_vec = job.shuffled_vector.back();
            job.shuffled_vector.pop_back();
            if (pthread_mutex_unlock(&job.lock_shuffled) != 0){
                system_error_handler(MUTEX_UNLOCK_ERROR);
            }
            auto reduce_context = new ReduceContext{&job.atomic_state, current_vec->size(), &job.lock_output, outputVec};
            client.reduce(current_vec, reduce_context);
            delete current_vec;
            delete reduce_context;
            if (pthread_mutex_lock(&job.lock_shuffled) != 0){
                system_error_handler(MUTEX_LOCK_ERROR);
            }
    }
        if (pthread_mutex_unlock(&job.lock_shuffled) != 0){
            system_error_handler(MUTEX_UNLOCK_ERROR);
        }

}

/**
 * Compares between two intermediate pairs
 * @param a the first pair
 * @param b the second pair
 * @return is the second pair bigger than the first pair
 */
bool cmp(const IntermediatePair &a, const IntermediatePair &b){
    return *(a.first)<*(b.first);
}

/**
 * The function that each thread runs. Contains all the four phases - map, sort, shuffle and reduce.
 * The only thread that can run the shuffle phase is thread 0.
 * @param arguments the arguments given to the thread
 * @return null pointer
 */
void *thread_routine(void* arguments){
    struct ThreadArgs *args = (ThreadArgs *)arguments;
    // MAP PHASE
    unsigned long int pair_num = (args->job)->map_partition_cnt.fetch_add(1);
    uint64_t state_value = (args->inputVec.size() << TOTAL_BIT) + MAP_STAGE;
    uint64_t helper = 0;
    args->job->atomic_state.compare_exchange_weak(helper, state_value);
    while (pair_num < args->inputVec.size()){
        InputPair current_pair = args->inputVec[pair_num];
        MapContext *map_context = new MapContext{&(args->job->mid_output[args->thread_num])};
        args->client.map(current_pair.first, current_pair.second, map_context);
        delete map_context;
        args->job->atomic_state += 1 << COUNTER_BIT;
        pair_num = args->job->map_partition_cnt.fetch_add(1);
    }
    std::sort(args->job->mid_output[args->thread_num].begin(),args->job->mid_output[args->thread_num].end(), cmp);
    auto current_mid_output = args->job->mid_output[args->thread_num];
    args->job->barrier->barrier();
    if (args->thread_num == 0){
        //SHUFFLE STAGE
        shuffle_phase(*(args->job));
        uint64_t state_value = (args->job->shuffle_total << TOTAL_BIT) + REDUCE_STAGE;
        args->job->atomic_state.exchange(state_value);
    }
    args->job->barrier->barrier();
    // REDUCE PHASE
    reduce_phase(args->client, *(args->job), &args->outputVec);
    delete args;
    return nullptr;
}


/**
 * This function starts running the MapReduce algorithm (with several threads) and returns a JobHandle.
 * @param client the client
 * @param inputVec input vector
 * @param outputVec the vector where the output should be saved
 * @param multiThreadLevel the number of worker threads to be used for running the algorithm.
 * @return JobHandle, contains the parameters which are relevant to the job
 */
JobHandle startMapReduceJob(const MapReduceClient& client,
                            const InputVec& inputVec, OutputVec& outputVec,
                            int multiThreadLevel){
    JobContext *job = new JobContext(multiThreadLevel);
    // create JobContext and initialize it:
    for (int i=0; i<multiThreadLevel; i++){
        auto *arguments = new ThreadArgs{i,client,inputVec,outputVec, job};
        if (pthread_create(&(job->threads[i]), nullptr, thread_routine, arguments) != 0) {
            system_error_handler(CREATE_THREAD_ERROR);
        }
    }
    return static_cast<JobHandle>(job);
}

/**
 * Gets JobHandle returned by startMapReduceFramework and waits until it is finished.
 * @param job a JobHandle object
 */
void waitForJob(JobHandle job){
    JobContext *job_context = (JobContext *)job;
    if (job_context->is_waitForJob_called){
        return;
    }
    job_context->is_waitForJob_called = 1;
    for (int i = 0; i < job_context->multiThreadLevel; ++i) {
        if (pthread_join(job_context->threads[i], nullptr) != 0){
            system_error_handler(JOIN_THREAD_ERROR);
        }
    }
}

/**
 * Gets a JobHandle and updates the state of the job into the given JobState struct
 * @param job a JobHandle object
 * @param state a struct where the current state should be saved
 */
void getJobState(JobHandle job, JobState* state){
    JobContext *job_context = (JobContext *)job;
    auto state_counter = job_context->atomic_state.load();
    auto stage = static_cast<stage_t>(state_counter & (0x3));
    float counter = state_counter>>COUNTER_BIT & (0x7fffffff);
    float total = state_counter>>TOTAL_BIT & (0x7fffffff);
     state->stage = stage;
     if(total == 0){
         state->percentage = 0;
         return;
     }
     state->percentage = (counter / total) * PERCENTAGE_FACTOR;
}

/**
 * Closes the given job, releases all its resources. After this function is called the job handle will be invalid. 
 * @param job the JobHandle object to close
 */
void closeJobHandle(JobHandle job){
    if(job == nullptr){
        return;
    }
    waitForJob(job);
    JobContext *job_context = (JobContext *)job;
    delete job_context;
}
