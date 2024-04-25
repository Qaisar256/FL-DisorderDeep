def initialize_parameters():
    """ Initialize the parameters and constants required for the computation. """
    params = {
        'memory_per_sample': 5,  # KB per sample
        'model_memory': 50 * 1024,  # MB to KB, converting model memory to KB
        'batch_size': 10,
        'epochs': 10,
        'time_per_sample_no_aug': 1,  # time in seconds per sample without data augmentation
        'time_per_sample_aug': 1.5  # time in seconds per sample with data augmentation
    }
    return params

def calculate_memory_and_time(samples, params):
    """ Calculate the total memory usage and training time with and without data augmentation. """
    # Calculating total samples with augmentation
    total_samples = sum(samples.values())

    # Total memory usage calculation with augmented data (assuming all data is loaded into memory for simplicity)
    total_memory_usage = total_samples * params['memory_per_sample'] + params['model_memory']  # in KB

    # Training time calculation
    total_time_no_aug = (total_samples / params['batch_size']) * params['epochs'] * params['time_per_sample_no_aug']  # in seconds
    total_time_aug = (total_samples / params['batch_size']) * params['epochs'] * params['time_per_sample_aug']  # in seconds

    return total_memory_usage, total_time_no_aug, total_time_aug

def main():
    # Initialize parameters
    params = initialize_parameters()

    # Placeholder dictionary for disease categories and their sample counts after data augmentation
    diseases = ['Healthy', 'COPD', 'Asthma', 'Pneumonia', 'URTI', 'Bronchiectasis', 'Bronchiolitis', 'LRTI']
    samples_augmented = {disease: 900 for disease in diseases}

    # Compute memory usage and training times
    total_memory_usage, total_time_no_aug, total_time_aug = calculate_memory_and_time(samples_augmented, params)

    # Output results
    print("Total Memory Usage (KB):", total_memory_usage)
    print("Total Training Time without Augmentation (s):", total_time_no_aug)
    print("Total Training Time with Augmentation (s):", total_time_aug)

if __name__ == "__main__":
    main()
