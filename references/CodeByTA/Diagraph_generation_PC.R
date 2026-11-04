# # Diagraph generation from PC algorithm.......................
library(readxl)   # call library 'readxl' to read .xlxs file 
library(pcalg)    # call library 'pcalg' for PC algorithm
library(openxlsx) # call library 'openxlsx' to open xlsx file
args <- commandArgs(trailingOnly = TRUE)
data_Normal_Path <- args[1]  # First argument is the file path for data
Fixed_Gaps_Path <- args[2]  # Second argument is the file path for Fixed Gaps
alpha1 <- as.numeric(args[3])   # second argument is the significance level
P<- as.numeric(args[4])  # third argument is the number of Tags
# data_Normal_Path <- "/home/brijesh/Desktop/code_to_send/MATLAB_CODE_without_analyzing_fault_event/Data_normal.xlsx" # First argument is the file path for data
# Fixed_Gaps_Path <- "/home/brijesh/Desktop/code_to_send/MATLAB_CODE_without_analyzing_fault_event/Fixed_gaps.xlsx"  # Second argument is the file path for Fixed Gaps
# alpha1 <- 0.05  # second argument is the significance level
# P<- 14  # third argument is the number of Tags

data <- read_xlsx(data_Normal_Path)  # read excel file of data sets. 
n<-nrow(data) # 'nrows' to find the number of rows in data
V<- paste0("X", seq_len(P)) # Automatically generate labels like "X1", "X2", "X3", etc.
FG_matrix <- read_xlsx(Fixed_Gaps_Path)  # read excel file to obtain Fixed Gaps matrix.
FG <- as.matrix(FG_matrix[1:P,2:P+1])
FG<-unname(FG)
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

