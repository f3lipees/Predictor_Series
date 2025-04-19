#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <signal.h>
#include <pthread.h>

#define MAX_SAMPLES 1000
#define WINDOW_SIZE 300
#define NUM_FEATURES 5
#define HIDDEN_LAYER_SIZE 10
#define OUTPUT_SIZE 1
#define LEARNING_RATE 0.01
#define PREDICTION_HORIZON 50
#define MAX_THREADS 4
#define OUTPUT_WIDTH 80
#define OUTPUT_HEIGHT 20

typedef struct {
    double **X;
    double *y;
    int n_samples;
    int n_features;
} Dataset;

typedef struct {
    double **weights1;
    double **weights2;
    double *bias1;
    double *bias2;
    double **hidden;
    double *output;
    int input_size;
    int hidden_size;
    int output_size;
} NeuralNetwork;

typedef struct {
    double *data;
    int size;
    int capacity;
    double min_val;
    double max_val;
} TimeSeries;

typedef struct {
    TimeSeries *original;
    TimeSeries *predicted;
    int running;
    pthread_mutex_t lock;
    char display[OUTPUT_HEIGHT][OUTPUT_WIDTH+1];
} Visualization;

typedef struct {
    Dataset *dataset;
    NeuralNetwork *model;
    TimeSeries *series;
    Visualization *viz;
    int start_idx;
    int end_idx;
    int thread_id;
} ThreadArgs;

volatile sig_atomic_t keep_running = 1;

double random_double() {
    return (double)rand() / RAND_MAX * 2.0 - 1.0;
}

void handle_sigint(int sig) {
    keep_running = 0;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double tanh_custom(double x) {
    return tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh(x);
    return 1.0 - t * t;
}

TimeSeries* create_time_series(int capacity) {
    TimeSeries *series = (TimeSeries*)malloc(sizeof(TimeSeries));
    series->data = (double*)calloc(capacity, sizeof(double));
    series->size = 0;
    series->capacity = capacity;
    series->min_val = INFINITY;
    series->max_val = -INFINITY;
    return series;
}

void add_time_series_value(TimeSeries *series, double value) {
    if (series->size >= series->capacity) {
        memmove(series->data, series->data + 1, (series->capacity - 1) * sizeof(double));
        series->size = series->capacity - 1;
    }

    series->data[series->size++] = value;

    if (value < series->min_val) {
        series->min_val = value;
    }
    if (value > series->max_val) {
        series->max_val = value;
    }
}

void free_time_series(TimeSeries *series) {
    free(series->data);
    free(series);
}

Dataset* create_dataset(int n_samples, int n_features) {
    Dataset *dataset = (Dataset*)malloc(sizeof(Dataset));
    dataset->n_samples = n_samples;
    dataset->n_features = n_features;

    dataset->X = (double**)malloc(n_samples * sizeof(double*));
    for (int i = 0; i < n_samples; i++) {
        dataset->X[i] = (double*)calloc(n_features, sizeof(double));
    }

    dataset->y = (double*)calloc(n_samples, sizeof(double));

    return dataset;
}

void free_dataset(Dataset *dataset) {
    for (int i = 0; i < dataset->n_samples; i++) {
        free(dataset->X[i]);
    }
    free(dataset->X);
    free(dataset->y);
    free(dataset);
}

void prepare_dataset_from_time_series(Dataset *dataset, TimeSeries *series, int window_size) {
    if (series->size < window_size + 1) return;

    int effective_size = series->size - window_size;
    if (effective_size > dataset->n_samples) effective_size = dataset->n_samples;

    for (int i = 0; i < effective_size; i++) {
        for (int j = 0; j < window_size; j++) {
            int feature_idx = j % dataset->n_features;
            dataset->X[i][feature_idx] = series->data[i + j];
        }
        dataset->y[i] = series->data[i + window_size];
    }
}

NeuralNetwork* create_neural_network(int input_size, int hidden_size, int output_size) {
    NeuralNetwork *nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->weights1 = (double**)malloc(input_size * sizeof(double*));
    for (int i = 0; i < input_size; i++) {
        nn->weights1[i] = (double*)malloc(hidden_size * sizeof(double));
        for (int j = 0; j < hidden_size; j++) {
            nn->weights1[i][j] = random_double() * sqrt(2.0 / input_size);
        }
    }

    nn->bias1 = (double*)calloc(hidden_size, sizeof(double));

    nn->weights2 = (double**)malloc(hidden_size * sizeof(double*));
    for (int i = 0; i < hidden_size; i++) {
        nn->weights2[i] = (double*)malloc(output_size * sizeof(double));
        for (int j = 0; j < output_size; j++) {
            nn->weights2[i][j] = random_double() * sqrt(2.0 / hidden_size);
        }
    }

    nn->bias2 = (double*)calloc(output_size, sizeof(double));

    nn->hidden = (double**)malloc(sizeof(double*));
    nn->hidden[0] = (double*)calloc(hidden_size, sizeof(double));

    nn->output = (double*)calloc(output_size, sizeof(double));

    return nn;
}

void free_neural_network(NeuralNetwork *nn) {
    for (int i = 0; i < nn->input_size; i++) {
        free(nn->weights1[i]);
    }
    free(nn->weights1);

    for (int i = 0; i < nn->hidden_size; i++) {
        free(nn->weights2[i]);
    }
    free(nn->weights2);

    free(nn->bias1);
    free(nn->bias2);
    free(nn->hidden[0]);
    free(nn->hidden);
    free(nn->output);
    free(nn);
}

void forward_propagation(NeuralNetwork *nn, double *input) {
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->hidden[0][i] = 0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->hidden[0][i] += input[j] * nn->weights1[j][i];
        }
        nn->hidden[0][i] += nn->bias1[i];
        nn->hidden[0][i] = tanh_custom(nn->hidden[0][i]);
    }

    for (int i = 0; i < nn->output_size; i++) {
        nn->output[i] = 0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->output[i] += nn->hidden[0][j] * nn->weights2[j][i];
        }
        nn->output[i] += nn->bias2[i];
    }
}

void backpropagation(NeuralNetwork *nn, double *input, double *target, double learning_rate) {
    forward_propagation(nn, input);

    double *output_error = (double*)malloc(nn->output_size * sizeof(double));
    double *hidden_error = (double*)malloc(nn->hidden_size * sizeof(double));

    for (int i = 0; i < nn->output_size; i++) {
        output_error[i] = target[i] - nn->output[i];
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        hidden_error[i] = 0;
        for (int j = 0; j < nn->output_size; j++) {
            hidden_error[i] += output_error[j] * nn->weights2[i][j];
        }
        hidden_error[i] *= (1 - nn->hidden[0][i] * nn->hidden[0][i]); // tanh derivative
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        for (int j = 0; j < nn->output_size; j++) {
            nn->weights2[i][j] += learning_rate * output_error[j] * nn->hidden[0][i];
        }
    }

    for (int i = 0; i < nn->output_size; i++) {
        nn->bias2[i] += learning_rate * output_error[i];
    }

    for (int i = 0; i < nn->input_size; i++) {
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->weights1[i][j] += learning_rate * hidden_error[j] * input[i];
        }
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        nn->bias1[i] += learning_rate * hidden_error[i];
    }

    free(output_error);
    free(hidden_error);
}

void *train_model_thread(void *args) {
    ThreadArgs *thread_args = (ThreadArgs*)args;
    Dataset *dataset = thread_args->dataset;
    NeuralNetwork *model = thread_args->model;
    int start_idx = thread_args->start_idx;
    int end_idx = thread_args->end_idx;

    for (int i = start_idx; i < end_idx && i < dataset->n_samples; i++) {
        backpropagation(model, dataset->X[i], &dataset->y[i], LEARNING_RATE);
    }

    return NULL;
}

void train_model(NeuralNetwork *model, Dataset *dataset, int epochs) {
    pthread_t threads[MAX_THREADS];
    ThreadArgs thread_args[MAX_THREADS];

    int samples_per_thread = dataset->n_samples / MAX_THREADS;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int t = 0; t < MAX_THREADS; t++) {
            thread_args[t].dataset = dataset;
            thread_args[t].model = model;
            thread_args[t].start_idx = t * samples_per_thread;
            thread_args[t].end_idx = (t == MAX_THREADS - 1) ? dataset->n_samples : (t + 1) * samples_per_thread;
            thread_args[t].thread_id = t;

            pthread_create(&threads[t], NULL, train_model_thread, &thread_args[t]);
        }

        for (int t = 0; t < MAX_THREADS; t++) {
            pthread_join(threads[t], NULL);
        }
    }
}

double predict(NeuralNetwork *model, double *input) {
    forward_propagation(model, input);
    return model->output[0];
}

void predict_future_values(NeuralNetwork *model, TimeSeries *original, TimeSeries *predicted, int horizon, int window_size) {
    if (original->size < window_size) return;

    double *input = (double*)malloc(NUM_FEATURES * sizeof(double));

    for (int i = 0; i < horizon; i++) {
        int start_idx = original->size - window_size + i;
        if (start_idx < 0) break;

        for (int j = 0; j < NUM_FEATURES; j++) {
            if (start_idx + j < original->size) {
                input[j] = original->data[start_idx + j];
            } else {
                int pred_idx = start_idx + j - original->size;
                if (pred_idx < predicted->size) {
                    input[j] = predicted->data[pred_idx];
                } else {
                    input[j] = 0;
                }
            }
        }

        double prediction = predict(model, input);
        add_time_series_value(predicted, prediction);
    }

    free(input);
}

Visualization* create_visualization(TimeSeries *original, TimeSeries *predicted) {
    Visualization *viz = (Visualization*)malloc(sizeof(Visualization));
    viz->original = original;
    viz->predicted = predicted;
    viz->running = 1;

    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            viz->display[i][j] = ' ';
        }
        viz->display[i][OUTPUT_WIDTH] = '\0';
    }

    pthread_mutex_init(&viz->lock, NULL);

    return viz;
}

void free_visualization(Visualization *viz) {
    pthread_mutex_destroy(&viz->lock);
    free(viz);
}

void render_console_chart(Visualization *viz) {
    double global_min = fmin(viz->original->min_val, viz->predicted->min_val);
    double global_max = fmax(viz->original->max_val, viz->predicted->max_val);
    double value_range = global_max - global_min;
    if (value_range == 0) value_range = 1;

    int original_size = viz->original->size;
    int predicted_size = viz->predicted->size;
    int total_points = original_size;

    pthread_mutex_lock(&viz->lock);

    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        for (int j = 0; j < OUTPUT_WIDTH; j++) {
            viz->display[i][j] = ' ';
        }
    }

    for (int j = 0; j < OUTPUT_WIDTH; j++) {
        viz->display[OUTPUT_HEIGHT - 1][j] = '-';
    }

    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        viz->display[i][0] = '|';
    }

    for (int i = 0; i < original_size && i < OUTPUT_WIDTH - 1; i++) {
        int x = i % (OUTPUT_WIDTH - 1) + 1;
        int y = OUTPUT_HEIGHT - 2 - (int)((viz->original->data[original_size - OUTPUT_WIDTH + i] - global_min) / value_range * (OUTPUT_HEIGHT - 2));
        if (y >= 0 && y < OUTPUT_HEIGHT) {
            viz->display[y][x] = 'o';
        }
    }

    for (int i = 0; i < predicted_size && i < OUTPUT_WIDTH - 1; i++) {
        int x = i % (OUTPUT_WIDTH - 1) + 1;
        int y = OUTPUT_HEIGHT - 2 - (int)((viz->predicted->data[i] - global_min) / value_range * (OUTPUT_HEIGHT - 2));
        if (y >= 0 && y < OUTPUT_HEIGHT) {
            viz->display[y][x] = 'x';
        }
    }

    pthread_mutex_unlock(&viz->lock);

    printf("\033[H\033[J"); // Clear the console (works on some terminals)
    printf("Time Series Prediction\n");
    printf("Original Series (o), Predictions (x)\n\n");

    for (int i = 0; i < OUTPUT_HEIGHT; i++) {
        printf("%s\n", viz->display[i]);
    }

    printf("\nCurrent data: %.6f\n", viz->original->data[viz->original->size - 1]);
    if (viz->predicted->size > 0) {
        printf("Next prediction: %.6f\n", viz->predicted->data[0]);
    }
    printf("Press Ctrl+C to exit\n");
}

void *visualization_thread(void *arg) {
    Visualization *viz = (Visualization*)arg;

    while (viz->running && keep_running) {
        render_console_chart(viz);
        usleep(200000); // Update every 200ms
    }

    return NULL;
}

double generate_synthetic_data(double time) {
    return sin(time * 0.1) + cos(time * 0.05) * sin(time * 0.02) + ((rand() % 100) - 50) / 500.0;
}

int main() {
    srand(time(NULL));
    signal(SIGINT, handle_sigint);

    TimeSeries *original_series = create_time_series(MAX_SAMPLES);
    TimeSeries *predicted_series = create_time_series(PREDICTION_HORIZON);

    Dataset *dataset = create_dataset(MAX_SAMPLES - WINDOW_SIZE, NUM_FEATURES);
    NeuralNetwork *model = create_neural_network(NUM_FEATURES, HIDDEN_LAYER_SIZE, OUTPUT_SIZE);

    Visualization *viz = create_visualization(original_series, predicted_series);

    pthread_t viz_thread;
    pthread_create(&viz_thread, NULL, visualization_thread, viz);

    FILE *csv_file = fopen("time_series_data.csv", "w");
    if (csv_file) {
        fprintf(csv_file, "timestamp,actual,predicted\n");
    }

    FILE *gnuplot_file = fopen("plot_script.gnuplot", "w");
    if (gnuplot_file) {
        fprintf(gnuplot_file, "set title 'Time Series Prediction'\n");
        fprintf(gnuplot_file, "set xlabel 'Time'\n");
        fprintf(gnuplot_file, "set ylabel 'Value'\n");
        fprintf(gnuplot_file, "set grid\n");
        fprintf(gnuplot_file, "set term png size 800,600\n");
        fprintf(gnuplot_file, "set output 'time_series_plot.png'\n");
        fprintf(gnuplot_file, "plot 'time_series_data.csv' using 1:2 with lines title 'Actual', ");
        fprintf(gnuplot_file, "'time_series_data.csv' using 1:3 with lines title 'Predicted'\n");
        fclose(gnuplot_file);
    }

    int count = 0;
    while (keep_running && viz->running) {
        double value = generate_synthetic_data(count);

        pthread_mutex_lock(&viz->lock);
        add_time_series_value(original_series, value);

        if (count % 10 == 0) {
            prepare_dataset_from_time_series(dataset, original_series, WINDOW_SIZE);
            train_model(model, dataset, 1);

            predicted_series->size = 0;
            predict_future_values(model, original_series, predicted_series, PREDICTION_HORIZON, WINDOW_SIZE);

            if (csv_file) {
                fprintf(csv_file, "%d,%.6f,%.6f\n",
                        count, value, predicted_series->size > 0 ? predicted_series->data[0] : 0.0);
                fflush(csv_file);
            }
        }
        pthread_mutex_unlock(&viz->lock);

        usleep(100000); // 100ms, i.e., 10 data points per second
        count++;
    }

    if (csv_file) {
        fclose(csv_file);
    }

    if (system("gnuplot plot_script.gnuplot") != 0) {
        printf("Note: Gnuplot visualization failed. Install gnuplot for graph generation.\n");
    }

    viz->running = 0;
    pthread_join(viz_thread, NULL);

    free_visualization(viz);
    free_neural_network(model);
    free_dataset(dataset);
    free_time_series(predicted_series);
    free_time_series(original_series);

    return 0;
}
