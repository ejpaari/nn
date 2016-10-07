"""
Clean away the 10% of points that have the largest
residual errors (difference between the prediction
and the actual net worth).

Return a list of tuples named cleaned_data where 
each tuple is of the form (age, net_worth, error).
"""
def outlier_cleaner(predictions, ages, net_worths):
    errors = (net_worths-predictions)**2
    zipped = zip(ages, net_worths, errors)
    cleaned = sorted(zipped, key = lambda k: k[2])
    return cleaned[:int(len(cleaned)*0.9)]

