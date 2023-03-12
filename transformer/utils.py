def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params:,} trainable parameters')