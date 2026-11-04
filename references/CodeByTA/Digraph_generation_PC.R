# # Diagraph generation from PC algorithm.......................
library(readxl)   # call library 'readxl' to read .xlxs file 
library(pcalg)    # call library 'pcalg' for PC algorithm
library(openxlsx)
library(readr)
# args <- commandArgs(trailingOnly = TRUE)
# data_Normal_Path <- args[1]  # First argument is the file path for data
# Fixed_Gaps_Path <- args[2]  # Second argument is the file path for Fixed Gaps
# alpha1 <- as.numeric(args[3])   # second argument is the significance level
# P<- as.numeric(args[4])  # third argument is the number of Tags

alpha1 <-0.05
P<- 14
data <- read_xlsx('/home/brijesh/Desktop/code_to_send/PYTHON_CODE/Data_normal.xlsx')
FG_matrix<- read_xlsx('/home/brijesh/Desktop/code_to_send/PYTHON_CODE/Fixed_gaps.xlsx')
# data <- read_xlsx(data_Normal_Path)  # provide the path for excel file of data sets.
# FG_matrix <- read_xlsx(Fixed_Gaps_Path)  # provide the path for excel file of data sets.
n<-nrow(data) # 'nrows' to find the number of rows in data
# Automatically generate labels like "V1", "V2", "V3", etc.
V<- paste0("X", seq_len(P))
# FG_matrix <- read_xlsx('/home/brijesh/Desktop/ingenero_25_11_24/PYTHON_CODE_copula_by_python_25nov_copy/Fixed_gaps.xlsx')  # read XLSX file to generate Fixed Gaps matrix
#FG_matrix <- read_csv('/home/brijesh/Desktop/ingenero_25_11_24/PYTHON_CODE_copula_by_python_25nov/Fixed_gaps.csv')  # read XLSX file to generate Fixed Gaps matrix
FG <- as.matrix(FG_matrix[1:P,2:P+1])
FG <- unname(FG)

# calling "pc" function for diagraph generation.....
pc.fit <- pc(suffStat = list(C = cor(data), n = n),
             indepTest = gaussCItest, ## indep.test: partial correlations
              alpha=alpha1,fixedEdges = NULL, fixedGaps =FG, labels=V, verbose = FALSE)

# for ploting R graph                         
if (require(Rgraphviz)) {  
  plot_file <- "PC_digraph.jpg"  # specify the file name
  jpeg(plot_file, width = 800, height = 600)
  plot(pc.fit, main = "Digraph from PC algorithm ") # plot the diagraph of data with 14 Tags
  dev.off()
  summary(pc.fit)   # summary obtained from pc algorithm.....
}  

##
# Assuming pc.fit is your PC algorithm output
adj_matrix <- as(pc.fit@graph, "matrix")  # Convert the graph to an adjacency matrix

# Print the adjacency matrix to confirm
print(adj_matrix)

# # Save the adjacency matrix to a CSV file
write.csv(adj_matrix, "adjacency_matrix.csv", row.names = FALSE)

# write.csv(adj_matrix, "adjacency_matrix.csv", rowNames = FALSE, colNames = FALSE)

# Optionally, save it as an R data file
#save(adj_matrix, file = "adjacency_matrix.RData")
# Write the matrix to an Excel file
# write.xlsx(adj_matrix, "adjacency_matrix.xlsx", rowNames = FALSE, colNames = FALSE)
# Check for symmetry


