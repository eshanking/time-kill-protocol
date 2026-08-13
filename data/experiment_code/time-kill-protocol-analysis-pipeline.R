library(data.table)
library(ggplot2)
library(openxlsx2)

setwd("C:/Users/s3ayy/Documents/GitHub/time-kill-protocol/")

data_folder <- "data/experiment_data/tk_Ecoli_CIP_07222026/"

time_values <- c(0, 15, 30, 45, 60, 90, 120, 180, 240, 300)

MIC <- 2^-5
#MIC <- 0.25
concentrations <- c(0, 8^-1, 8^0, 8^1, 8^2, 8^3) * MIC

# Need to get all files

file_list <- list.files(path = data_folder, full.names = TRUE)

# For each file
# time at E7
RFU_summary_list <- list()
i <- 1
for (i in 1:length(file_list)) {
  file_name <- file_list[i]
  
  for (concentration_category in 1:3) {
    
    time_string <- wb_read(file_name, sheet = concentration_category, rows = 6:7, cols = 5:5)[[1]]
    hour <- as.numeric(strsplit(time_string, split = ":")[[1]][1])
    minutes <- as.numeric(strsplit(strsplit(time_string, split = " ")[[1]][1], split = ":")[[1]][2])
    day_half <- strsplit(time_string, split = " ")[[1]][2]
    
    if (day_half == "AM") {
      absolute_time <- (hour %% 12) * 60 + minutes
    } else if (day_half == "PM") {
      absolute_time <- (hour %% 12) * 60 + minutes + 12 * 60
    }
    
    data <- as.matrix(wb_read(file_name, sheet = concentration_category, rows = 53:59, cols = 3:12))
    colnames(data) <- NULL
    rownames(data) <- NULL

    if (any(is.na(data))) {
      print("NA Values in loaded matrix!!")
      stop()
    }
    
    current_column <- data[,i]
    concentration_vector <- c(rep(concentrations[concentration_category *2 - 1],3), rep(concentrations[concentration_category *2],3))
    
    #current_time <- time_values[i]
    current_time <- absolute_time
    
    time_vector <- rep(current_time, 6)
    rows <- c("B", "C", "D", "E", "F", "G")
    
    data_table_tmp <- data.table(RFU = current_column, concentration = concentration_vector, time = time_vector, row = rows)
    
    RFU_summary_list[[3*(i - 1) + concentration_category]] <- data_table_tmp
  }
}

RFU_summary_dt <- rbindlist(RFU_summary_list)

RFU_summary_dt$time <- RFU_summary_dt$time - min(RFU_summary_dt$time)

ggplot(data = RFU_summary_dt, aes(x = time, y = RFU, colour = row)) +
  geom_point() + facet_wrap(~concentration) + scale_y_log10() +
  coord_cartesian(ylim = c(100, 10000))



# Split by underscore, take last value remove .xlsx
# Read that column of the plate and add it to the summary data table (Time, Concentration, Row)


library(patchwork)

p_Ecoli + coord_cartesian(ylim = c(1000, 10000)) +
  p_Saureus + coord_cartesian(ylim = c(1000, 10000))




# Acting on the summary data frame, plot the values on a log y, facet wrap by concentration + row

