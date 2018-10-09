

# ## 2. Preprocess each subject
#
# Here we prepare the dataframe for analysis:
# 1. Create a hierarchical index:
#     - sup_index: subject id (sid) converted from raw **participant:assignmentId**
#     - sub_index: trial number
# 2. Add a new column encoding switch trials
# 3. Reformat the **blockTrial** column to be the number of trials played in the new block (simply subtract 1 from the column)
# 4. Parse the **monster** column into two columns encoding dimensions 1 and 2
# 5. Convert non-numerical values to numbers:
#     - **category** (`str` to `int`):
#         - category1D -> 1
#         - categoryIgnore1D -> 2
#         - category2D -> 3
#         - categoryRandom -> 4
#     - **family** (`str` to `int`):
#         - Bear -> 1
#         - Bunny -> 2
#         - GreenMonster -> 3
#         - Squid -> 4
#     - **state** (`str` to `int`):
#         - train -> 0
#         - free -> 1
#         - test -> 2
#     - **correct** (`bool` to `int`):
#         - False -> 0
#         - True -> 1
# 6. Clean up by removing unwanted columns
#
# Finally we rearrange the columns for better presentation:
# - columns 1 to 3 contain experiment variables: condition, state, blockTrial
# - columns 2 to 5 contain stimulus variables: family, dimension 1, dimension 2, category (difficulty)
# - columns 6 to 8 contain response variables: correct, switch, rt
#
# The resulting data is exclusively numeric, so conversion into numpy array for calculations is trivial.

# In[3]:
