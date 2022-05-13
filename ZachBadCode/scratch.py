sums = 2 + 2 + 6 + 2 + 6 + 1 + 2 + 6 + 9 + 1 + 40 + 4 + 1 + 40 + 1 + 10 + 1 + 4 + 8 + 5 + 1 + 9 + 1 + 9 + 7 + 1 + 13 + 65 + 59 + 40 + 3 + 7 + 6 + 5 + 11 + 8 + 3 + 8 + 18 + 24 + 24 + 9 + 9  + 3 + 5 + 8 + 3 + 6 + 8
print(sums)




# dead code:

# def original_entry_keys(data, visualize=False, large_cols_names=True):
#     '''
#     For each column, show each orignial entry,
#     as well as how many times that entry appears. 
#     '''
#     key_data = []
#     for col in data:
#         keys = [i for i in data[col].value_counts(dropna=False).keys()]
#         key_data.append(keys)
#         if visualize:
#             # print(len(data[col].value_counts(dropna=False)))
#             # print(data[col].value_counts(dropna=False)[0:10])
#             # print()
#         if large_cols_names:
#             if len(data[col].value_counts(dropna=False)) > 70:
                
#     return key_data