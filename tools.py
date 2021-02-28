
def print_results(test_set_labels, tp, fp):
    data_scientists = len(test_set_labels.index)
    wants_to_change = test_set_labels.sum()
    # does_not_want = data_scientists - wants_to_change
    percentage = round(wants_to_change / data_scientists * 100, 1)

    print(f"""
    Without ML model:
    {data_scientists} calls to data scientists, out of which 
    {percentage} % would be beneficial, resulting in 
    {wants_to_change} data scientists changing jobs.""")

    to_call = tp + fp
    beneficial = round(tp / to_call * 100, 1)
    found_ds = tp
    found_percentage = round(found_ds / wants_to_change * 100, 1)
    called_percentage = round(to_call / data_scientists * 100, 1)

    print(f"""
    With ML model:
    {to_call} calls to data scientists, out of which
    {beneficial} % would be beneficial, resulting in
    {found_ds} data scientists changing jobs.
    
    Call {called_percentage} % of data scientists 
    -> find {found_percentage} % of the ones willing to change""")
