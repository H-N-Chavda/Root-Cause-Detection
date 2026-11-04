
# pandas.read_excel to read Excel files.
# Graph Operations: Used networkx to find cycles in
# Use NumPy for operations like mean, standard deviation, covariance, and matrix multiplication.

# =============================================================================

# Import custom functions from the functions directory
from function_file.tsq_stats import Tsq_stats
from function_file.tsq_decomp import Tsq_decomp
from function_file.plot_star import plot_star
from function_file.Plot_Raw_Tsq import Plot_Raw_Tsq
from function_file.Hist_plot import Hist_plot
from function_file.copula_transformed import copula_transformed
from function_file.NAN_remove import NAN_remove
#==============================================================================

# Importing necessary libraries
import subprocess
import networkx as nx
from scipy.stats import f
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import numpy as np
import os
print('All import completed')
#==============================================================================

# Set the path to your R installation (ensure this is the correct R path)
os.environ['R_HOME'] = '/usr/lib/R' # In Linex
#os.environ['R_HOME'] = 'C:/Program Files/R/R-4.4.1'  # In Window
#==============================================================================

start_time = time.time() # to compute the computational time

# Input argument
print('Enter the mode in which you want to run the code: ')
print('1: Without transformation ')
print('2: With Copula transformation ')
Option = int(input("Enter your Choice: "))
print('Load all variables completed')

# Enter significance level (alpha value)
alpha1 = float(input("Enter the significance level (alpha) value: "))  # 0.05

# read the Excel file  of oprational data (i.e. Complete_data_minutewise.xlsx)
data_df = pd.read_excel('Complete_data_minutewise.xlsx', header=0)

# read the Excel file  of User input (i.e. User_input.xlsx)
user_input_df = pd.read_excel('User_input.xlsx')  # Assuming user input with 'Tags', 'Min_value', and 'Max_value'

# Extract column names from user_input
column_names_list = user_input_df['Tags'].dropna().tolist()  # Drop NaN values to avoid errors

# Ensure column names in data_df and user_input match by cleaning up whitespaces
data_df.columns = data_df.columns.str.strip()
column_names_list = [col.strip() for col in column_names_list]

# Identify missing columns and notify the user
missing_columns = [col for col in column_names_list if col not in data_df.columns]
if missing_columns:
    print(f"The following columns are missing in data_df and will be skipped: {missing_columns}")

# Filter data_df to retain only the relevant columns from user_input
valid_columns = [col for col in column_names_list if col in data_df.columns]
filtered_df = data_df[valid_columns]

# Extract min and max values for each column from user_input
column_ranges = user_input_df.set_index('Tags')[['Min_value', 'Max_value']].to_dict(orient='index')
#cleaned_column_ranges = {key: value for key, value in column_ranges.items() if not (pd.isna(value['Min_value']) and pd.isna(value['Max_value']))}

for column in column_names_list:
    if column in column_ranges:
        min_val = column_ranges[column].get('Min_value', float('nan'))
        max_val = column_ranges[column].get('Max_value', float('nan'))

        # Apply lower bound filter
        if pd.notna(min_val):  # Apply lower bound filter only if min_val is not NaN
            filtered_df[column] = filtered_df[column].clip(lower=min_val)

        # Apply upper bound filter
        if pd.notna(max_val):  # Apply upper bound filter only if max_val is not NaN
            filtered_df[column] = filtered_df[column].clip(upper=max_val)


Data_abnormal = filtered_df.values


# # # # Convert to a pandas DataFrame for easier processing (pandas handles mixed data better)
df = pd.DataFrame(Data_abnormal)
# # # # Use pd.to_numeric with 'coerce' option to convert to numeric and handle errors
# # # # If a value cannot be converted to numeric, it will be replaced by NaN
df_numeric = df.apply(pd.to_numeric, errors='coerce')

# # # # Now we want to remove the rows with NaN values, as they correspond to non-numeric text
df_clean = df_numeric.dropna(how='all')  # Drop rows where all values are NaN

# # # # Convert back to numpy array (matrix with only numeric values)
Data = df_clean.to_numpy()

P =  Data.shape[1]
# function call for NAN removal
Data = NAN_remove(Data)
# To convert the unit from TPH to Kmol/h
Data[:, 0]=Data[:, 0] * Data[:, 4]
Data[:, 1]=Data[:, 1] * Data[:, 4]

Data_abnormal=Data

# Create Normal data (slippage event free)
Data_normal = Data_abnormal[Data_abnormal[:, P-1] <= 1]
# #..................Copula transformation for all data...............................................
if Option == 2:
    # Compute the standard deviation for each column
    std_Data_normal =np.std(Data_normal, axis=0,ddof=1)
    
    # Check if there are any zero standard deviations and store the index 
    zero_std_columns = np.where(std_Data_normal == 0)[0]
    
    # If there are any zero standard deviations
    if zero_std_columns.size == 0:
        Tags_error_normal = []
    else:
        Tags_error_normal = zero_std_columns

    # Perform Copula transformation on the data
    Data_normal = copula_transformed(Data_normal, Tags_error_normal)
    print('Copula transformation for all data completed')

else:
    print('Without transformation for all data completed')
# .........................................end..............Copula transformation...........................................


# Convert the matrix into a pandas DataFrame
df = pd.DataFrame(Data_normal)

# Write the DataFrame to an Excel file
df.to_excel('Data_normal.xlsx', index=False, header=False)

# To obtain the current directory
current_directory = os.getcwd()

# Load information of restriction on edges in excel file (enter in
# symmetrical manner)(i.e. 1-2 node is absent then [1,2]=[2,1]=1)

# Define the paths to the Excel files (i.e Fixed_gaps.xlsx)
FG_path = os.path.join(current_directory, 'Fixed_gaps.xlsx')

# Define the paths to the Excel files (i.e Data_normal.xlsx)
Data_normal_path = os.path.join(current_directory, 'Data_normal.xlsx')

# #================== Calling R studio to obtain Digraph and  adjacency_matrix====================================
# Define the path to your R script
r_script = 'Digraph_generation_PC.R'

# Construct the command string
command = f'Rscript {r_script} {Data_normal_path} {FG_path} {alpha1:.2f} {P:.2f}'

# Run the command using subprocess to call R
result=subprocess.run(command, shell=True, text=True, capture_output=True)
img = mpimg.imread('PC_digraph.jpg')

# Display the image in Python
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.show()
print(result.stdout)

# Note: The digraph and adjacency_matrix are saved in current diectory with name PC_digraph.jpg and adjacency_matrix.csv respectively

# ======================================================
# Read the adjacency matrix from the CSV file
adj_matrix = pd.read_csv('adjacency_matrix.csv')

# Create numpy array for adj_matrix:
adj_matrix = adj_matrix.to_numpy()


# Create a directed graph from the adjacency matrix to check the cycle in digraph
G = nx.DiGraph()

# Add edges from the adjacency matrix
n = adj_matrix.shape[0]  # number of nodes

for i in range(n):
    for j in range(n):
        if adj_matrix[i, j] != 0:  # If there is an edge from node i to node j
            G.add_edge(i, j)

# Find all cycles in the graph
cycles = list(nx.simple_cycles(G))


if len(cycles) == 0:
    print("Digraph has no cycle")
    
    # Create matrix with all zeroes element to store the statistics
    Tsq_slip_cross_thrs = np.zeros((len(Data_abnormal), P + 1)) # Slippage occure and the respective variable crosses the threshold (i.e. both event takes place)
    Tsq_all = np.zeros((len(Data_abnormal), P + 1))  # Store the Tsq proofile of all the Tags
    Tsq_star = np.zeros((len(Data_abnormal), P + 1))  # Slippage occure and all the respective variable crosses the threshold which correspond to store the i^th value for i^th Tags
    Tsq_slip = np.zeros((len(Data_abnormal), P + 1))  # Slippage occure and then store the profile of each variables
    faulty_store = np.zeros((len(Data_abnormal), 1))  # Slippage occure and then store the profile of each variables

    WL = 50 * P  # Window length

    # #------------------------------------------- Static window (First window)--------------------------------------
    # Create abnormal data in a window (data with slippage event)
    Window_slip = Data_abnormal[:WL, :]
    
    # Create Normal data (slippage event free)
    Window_normal = Window_slip[Window_slip[:, P-1] <= 1]
    Tags_error_slip = []
    Tags_error_normal = []
    std_Mat = np.std(Window_normal, axis=0,ddof=1)
    
    if np.any(std_Mat == 0):
       faulty_store[:WL, 0] = P + 2
    else:
        
    # ....................Copula transformation in static window for abnormal and normal data..............................................
        if Option == 2:
            Window_normal = copula_transformed(Window_normal, Tags_error_normal)
            Window_slip = copula_transformed(Window_slip, Tags_error_slip)
    # ....................................................................................................

        mean_Mat = np.mean(Window_normal, axis=0)
        std_Mat = np.std(Window_normal, axis=0,ddof=1)
        Cov_mat = np.corrcoef(Window_normal, rowvar=False)
    
        # Normalized faulty data
        Mat_normalized = np.zeros((WL, P))
    
        # Perform the normalization
        for i in range(P):  # Loop through each column
           for j in range(WL):  # Loop through each row
              Mat_normalized[j, i] = (Window_slip[j, i] - mean_Mat[i]) / std_Mat[i]
    
        
        # Compute Tsq_sample as the diagonal of the matrix product
        Tsq_sample = np.diagonal(np.dot(np.dot(Mat_normalized, np.linalg.pinv(Cov_mat)), Mat_normalized.T))
    
        # Call function Tsq_decomp to obtain Tsq decomposition of each variables
        Tsq, Lcond = Tsq_decomp(adj_matrix, Mat_normalized, Cov_mat,Tags_error_normal)
        
        #Initialize the threshold vector of each tags equal to zeroes
        Tsq_thrs = np.zeros(P+1)
    
        # Threshold computation of each variables 
        p = 1 #(i.e. for each individual tags)
        for nodei in range(P):
            Tsq_thrs[nodei] = (((WL + p) * (WL - p)) / (WL * (WL - Lcond[nodei] - p))
                               ) * f.ppf(1 - alpha1, p, WL - Lcond[nodei] - p)
            
        #Threshold computation for Tsq sample (i.e. all variables)
        Tsq_thrs[P] = (((WL + P) * (WL - P)) / (WL * (WL - P))) * \
        f.ppf(1 - alpha1, P, WL - P)
    
        for i1 in range(WL):
            for nodei in range(P):
                   Tsq_all[i1, nodei] = Tsq[nodei][i1]
    
            Tsq_all[i1, P] = Tsq_sample[i1]
    
            if Data_abnormal[i1, P-1] > 1: #slippage check
    
                for nodei in range(P):
                    Tsq_slip[i1, nodei] = Tsq[nodei][i1]
                    
                    if Tsq[nodei][i1] > Tsq_thrs[nodei]:
                        Tsq_slip_cross_thrs[i1, nodei] = Tsq[nodei][i1]
                        Tsq_star[i1, nodei] = nodei+1
    
                if Tsq_sample[i1] > Tsq_thrs[P]:
                    Tsq_slip_cross_thrs[i1, P-1] = Tsq_sample[i1]
                    Tsq_star[i1, P] = P+1
   
    # ======================================Moving window============================================================
    for k in range(1, len(Data_abnormal) - (WL-1)):
        # k=69529
        print(k)
        
        # Create abnormal data in a window (data with slippage event)
        Window_slip = Data_abnormal[k:WL + k, :]
       
        # Create Normal data (slippage event free)
        Window_normal = Window_slip[Window_slip[:, P-1] <= 1]
        
        std_Mat = np.std(Window_normal, axis=0,ddof=1)
        
        # Check if any element of std_Mat is zero
        if np.any(std_Mat == 0):
            faulty_store[WL - 1 + k, 0]=P+2
        else:
        #==============================================Copula transformation==================================
            if Option == 2:
                 Window_slip = copula_transformed(Window_slip,Tags_error_slip)
                 Window_normal = copula_transformed(Window_normal,Tags_error_normal)
             
            mean_Mat = np.mean(Window_normal, axis=0)
            std_Mat = np.std(Window_normal, axis=0,ddof=1)
            Cov_mat = np.corrcoef(Window_normal, rowvar=False)
    
            # Normalized faulty data
            Mat_normalized = np.zeros((WL, P))
    
            # Perform the normalization
            for i in range(P):  # Loop through each column
               for j in range(WL):  # Loop through each row
                  Mat_normalized[j, i] = (Window_slip[j, i] - mean_Mat[i]) / std_Mat[i]
            
            # Compute Tsq_sample as the diagonal of the matrix product
            Tsq_sample = np.diagonal(np.dot(np.dot(Mat_normalized, np.linalg.pinv(Cov_mat)), Mat_normalized.T))
    
    
            # Call function Tsq_decomp to obtain Tsq decomposition of each variables
            Tsq, Lcond = Tsq_decomp(adj_matrix, Mat_normalized, Cov_mat,Tags_error_normal)
    
            for nodei in range(P):
                Tsq_all[WL - 1 + k, nodei] = Tsq[nodei][-1]
    
            Tsq_all[WL - 1 + k, P] = Tsq_sample[-1]
    
            if Data_abnormal[WL - 1 + k, P-1] > 1:
                for nodei in range(P):
    
                    Tsq_slip[WL - 1 + k, nodei] = Tsq[nodei][-1]
                    if Tsq[nodei][-1] > Tsq_thrs[nodei]:
                        Tsq_slip_cross_thrs[WL - 1 + k, nodei] = Tsq[nodei][-1]
                        Tsq_star[WL - 1 + k, nodei] = nodei+1
    
                if Tsq_sample[-1] > Tsq_thrs[P]:
                    Tsq_slip_cross_thrs[WL - 1 + k, P] = Tsq_sample[-1]
                    Tsq_star[WL - 1 + k, P] = P+1
    
    # Define the range for plotting (Case-1)
    range_plot = [1, len(Data_abnormal)]
    # Function file for the plot of the raw data as well as the Tsq plot
    # Plot_Raw_Tsq is a custom function, and we call it here assuming it's defined elsewhere
    X1 = Plot_Raw_Tsq(Data_abnormal, Tsq_slip, Tsq_all, Tsq_thrs, range_plot)

    # # Function file for the start plot
    # # Plot_Star is a custom function, and we call it here assuming it's defined elsewhere
    X1 = plot_star(Tsq_star, range_plot)

    # # Function file for the histogram plot
    # # Hist_plot is a custom function, and we call it here assuming it's defined elsewhere
    X3 = Hist_plot(X1, P)

    # # Function file for calculating explained, unexplained, and partially explained cases
    # Tsq_stats is a custom function, and we call it here assuming it's defined elsewhere
    Count = Tsq_stats(Data_abnormal, Tsq_star, faulty_store,range_plot, P)

    print(f'No. of times slippage occur: {Count[3]}')
    print(f'No. of times slippage events detected: {Count[4]}')
    print(f'Fully explained case: {Count[0]}')
    print(f'Partially explained case: {Count[1]}')
    print(f'Unexplained case: {Count[2]}')
    print(f'Uncaptured case: {Count[5]}')
    print(f'Faulty case: {Count[6]}')                

else:
    print('Digraph has cycle: Construct New Adjancy matrix such that to account cycle')


#==================================Window things are completed============================================
#========================Now analyze the data=============================================================



end_time = time.time()

# Compute and print the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed Time: {elapsed_time} seconds")
