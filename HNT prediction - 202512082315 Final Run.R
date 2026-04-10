# ==============================================================================
# INT6133 - Project HNT Analysis Code (Integrated Hybrid Approach)
# Purpose: 
#   1. Apply Feature Engineering globally (Neighbors + HK)
#   2. Train Physics Proxy using Engineered Features
#   3. Feed Physics Output (Pred_GroundTemp) into Supervised Models
# ==============================================================================

# 1. Setup Libraries -----------------------------------------------------------
pkgs <- c("caret", "randomForest", "C50", "klaR", "e1071", "plyr", "dplyr", "readr", "zoo", "pROC", "xgboost")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

library(caret)
library(C50)
library(klaR)
library(dplyr)
library(readr)
library(zoo)
library(pROC)
library(xgboost)

# 2. Robust Helper Functions ---------------------------------------------------

force_numeric <- function(x) {
  if(is.numeric(x)) return(x)
  x_clean <- gsub("[^0-9.-]", "", as.character(x))
  x_clean[x_clean == ""] <- NA
  return(suppressWarnings(as.numeric(x_clean)))
}

robust_read_csv <- function(filepath) {
  encodings <- c("GB18030", "UTF-8", "Big5", "GBK")
  for(enc in encodings) {
    tryCatch({
      df <- read_csv(filepath, locale = locale(encoding = enc), 
                     col_types = cols(.default = "c"), show_col_types = FALSE)
      if(nrow(df) > 0) return(df)
    }, error = function(e) {})
  }
  stop(paste("Failed to read:", filepath))
}

rename_cols <- function(df) {
  if (!is.data.frame(df)) return(NULL)
  df_cols <- colnames(df)
  mapping <- list(
    City = c("地名", "City"), Date = c("日期", "Date"),
    Lat = c("緯度", "纬度", "Lat"), Lon = c("經度", "经度", "Lon"),
    AirTemp = c("日平均氣溫", "日平均气温", "AirTemp"),
    GroundTemp = c("地表温度", "地表", "地温", "GroundTemp"), 
    DewPoint = c("日平均露點溫度", "日平均露点", "露點", "露点", "DewPoint"),
    Humidity = c("日平均相對濕度", "相对湿度", "相對濕度", "湿度", "Humidity"),
    Pressure = c("日平均氣壓", "日平均气压", "氣壓", "气压", "Pressure"),
    WindSpeed = c("平均風速", "平均风速", "風速", "风速", "WindSpeed"),
    WindDir = c("盛行風向", "盛行风向", "風向", "风向", "WindDir"), 
    Radiation = c("太陽總輻射", "太阳总辐射", "輻射", "辐射", "Radiation"),
    Cloud = c("日平均雲量", "日平均云量", "雲量", "云量", "Cloud"),
    HNT_Label = "Event_Occurred"
  )
  for(new_name in names(mapping)) {
    keywords <- as.character(mapping[[new_name]])
    for(key in keywords) {
      idx <- grep(key, df_cols, fixed = TRUE)
      if(length(idx) > 0) { colnames(df)[idx[1]] <- new_name; break }
    }
  }
  return(df)
}

numeric_cols <- c("AirTemp", "GroundTemp", "DewPoint", "Humidity", "Pressure", 
                  "WindSpeed", "WindDir", "Radiation", "Cloud", "Lat", "Lon")

# 3. GLOBAL FEATURE ENGINEERING FUNCTION ---------------------------------------
# This function applies the logic to ANY dataset (Neighbors or HK)
# Crucial: It respects grouping by City to prevent lag errors across datasets

engineer_features <- function(df) {
  # 1. Ensure Types
  df <- df %>% mutate(across(any_of(numeric_cols), force_numeric))
  
  # 2. Sort
  if("Date" %in% colnames(df)) df <- df %>% arrange(Date)
  
  # 3. Calculations
  df <- df %>% mutate(
    # A. Monsoon Logic (Cosine Transform)
    # Default to 0 (Neutral) if WindDir missing
    Wind_NorthSouth = if("WindDir" %in% colnames(.)) cos(WindDir * pi / 180) else 0,
    
    # B. Dynamics (Rate of Change)
    DewPoint_Change = DewPoint - lag(DewPoint, 1),

    # C. Thermal Inertia (Lags & Rolling)
    AirTemp_Lag1 = lag(AirTemp, 1),
    AirTemp_Lag2 = lag(AirTemp, 2),
    DewPoint_Roll3 = rollmean(DewPoint, k = 3, fill = NA, align = "right"),
    
    # D. Physics Interactions
    # Note: We cannot calculate Delta_T here yet for HK, because HK lacks GroundTemp.
    # We will add Pred_GroundTemp later.
    Dryness_Power = (AirTemp - DewPoint) * WindSpeed
  )
  return(df)
}

# 4. Data Loading & Processing -------------------------------------------------

# A. Neighbor Data
neighbor_files <- c("深圳15-24年天气数据_对齐.csv", "湛江15-24年天气数据_对齐.csv",
                    "珠海15-24年天气数据_对齐.csv", "茂名15-24年天气数据_对齐.csv",
                    "阳江15-24年天气数据_对齐.csv", "汕头15-24年天气数据_对齐.csv",
                    "汕尾15-24年天气数据_对齐.csv")

neighbors_list <- list()
print("--- Loading Neighbor Data ---")
for (f in neighbor_files) {
  if(file.exists(f)) {
    temp <- robust_read_csv(f)
    temp <- rename_cols(temp)
    # APPLY ENGINEERING PER CITY
    temp <- engineer_features(temp) 
    neighbors_list[[f]] <- temp
  }
}
neighbors_data <- bind_rows(neighbors_list)

# B. Hong Kong Data
hk_file <- "香港15-24年天气数据_回南天_Labelled.csv"
print("--- Loading Hong Kong Data ---")
if(!file.exists(hk_file)) stop("HK Data Missing")

hk_data <- robust_read_csv(hk_file)
hk_data <- rename_cols(hk_data)
# APPLY ENGINEERING TO HK
hk_data <- engineer_features(hk_data)

if("HNT_Label" %in% colnames(hk_data)) {
  hk_data$HNT_Label <- as.factor(ifelse(hk_data$HNT_Label == 1, "Yes", "No"))
}

# ==============================================================================
# PART 5: PHYSICS PROXY (Enhanced with Engineered Features)
# ==============================================================================
print("--- Training Physics Proxy on Neighbors (Using Engineered Features) ---")

# We now use the engineered features (Lags, Wind_NorthSouth) to predict GroundTemp
phys_predictors <- c("AirTemp", "DewPoint", "Humidity", "Pressure", 
                     "WindSpeed", "Wind_NorthSouth", "Radiation", "Cloud",
                     "DewPoint_Change", "Dryness_Power",
                     "DewPoint_Roll3", "AirTemp_Lag1", "AirTemp_Lag2")

# Intersect to find what's actually available
phys_vars <- intersect(phys_predictors, colnames(neighbors_data))
model_data_neigh <- neighbors_data %>% dplyr::select(all_of(c("GroundTemp", phys_vars))) %>% na.omit()

# Train Model
phys_model <- lm(GroundTemp ~ ., data = model_data_neigh)
print(summary(phys_model))

# Predict on HK
print("--- Predicting Hong Kong Ground Temperature ---")
hk_input <- hk_data %>% dplyr::select(any_of(phys_vars))
# Impute NAs (from lags) with mean to prevent prediction loss
hk_input[] <- lapply(hk_input, function(x) ifelse(is.na(x), mean(x, na.rm=TRUE), x))

hk_data$Pred_GroundTemp <- predict(phys_model, hk_input)

# Calculate the critical physics metric based on PREDICTION
hk_data$Phys_Delta_T <- hk_data$DewPoint - hk_data$Pred_GroundTemp


# ==============================================================================
# PART 6: SUPERVISED LEARNING (Hybrid Approach)
# ==============================================================================
print("--- Training Supervised Models (Hybrid: Including Pred_GroundTemp) ---")

# Define features for AI (Including the new Physics Proxy outputs)
ai_features <- c(phys_vars, "Pred_GroundTemp")
ai_features <- intersect(ai_features, colnames(hk_data))

print(ai_features)

# Clean Data
if(length(unique(hk_data$HNT_Label)) < 2) stop("One class only in target!")
hk_clean <- hk_data %>% dplyr::select(all_of(ai_features), HNT_Label) %>% na.omit()

# Split
set.seed(2025)
train_idx <- createDataPartition(hk_clean$HNT_Label, p = 0.7, list = FALSE)
hk_train <- hk_clean[train_idx, ]
hk_test  <- hk_clean[-train_idx, ]

# Setup Control (Up-Sampling for Imbalance)
ctrl <- trainControl(method = "cv", number = 5, sampling = "up", classProbs = TRUE)

# 1. C5.0
print("Training C5.0 (Hybrid)...")
grid_c50 <- expand.grid(.model = "tree", .trials = c(1, 10), .winnow = FALSE)
mod_c50 <- train(HNT_Label ~ ., data = hk_train, method = "C5.0", 
                 tuneGrid = grid_c50, trControl = ctrl, metric = "Kappa", na.action = na.omit)

# 2. Naive Bayes
print("Training Naive Bayes (Hybrid)...")
# Force Kernel estimation (flexible distribution) and add smoothing
# grid_nb <- expand.grid(usekernel = c(TRUE, FALSE), fL = c(0, 1), adjust = c(1, 1.5))
# mod_nb <- train(HNT_Label ~ ., data = hk_train, method = "nb", 
#                 tuneGrid = grid_nb, trControl = ctrl, metric = "Kappa", na.action = na.omit)

# 1. REMOVE fL = 0 (This is the most critical fix). 
#    fL = 1 adds a small "smoothing" value so no probability is ever exactly zero.
# 2. REMOVE usekernel = FALSE. 
#    TRUE allows the model to fit "weird" shapes, not just perfect bell curves.

grid_nb <- expand.grid(
  usekernel = c(TRUE), 
  fL = c(1),        
  adjust = c(2.0)
)

mod_nb <- train(
  HNT_Label ~ ., 
  data = hk_train, 
  method = "nb", 
  tuneGrid = grid_nb, 
  trControl = ctrl, 
  metric = "Kappa", 
  na.action = na.omit,
  preProcess = c("zv", "nzv", "center", "scale", "pca")
)


# 3. Logistic Regression
print("Training GLM (Hybrid)...")
mod_glm <- train(HNT_Label ~ ., data = hk_train, method = "glm", family = "binomial",
                 trControl = ctrl, metric = "Kappa", na.action = na.omit)

# ==============================================================================
# PART 7: EVALUATION & SCORECARD (Corrected & Enhanced)
# ==============================================================================

# 1. Helper Functions ----------------------------------------------------------

# A. Advanced Evaluation Function (For Probabilistic AI Models)
# Returns: List containing metrics and best Threshold
evaluate_model <- function(model_name, truth, probs, plot_roc = TRUE) {
  
  # 1. Calculate ROC & AUC
  roc_obj <- roc(response = truth, predictor = probs, levels = c("No", "Yes"), quiet = TRUE)
  auc_val <- as.numeric(roc_obj$auc)
  
  # 2. Find Best Threshold (Youden's J)
  best_coords <- coords(roc_obj, "best", best.method = "youden", ret = "threshold", transpose = FALSE)
  
  # Handle edge case if multiple thresholds match (take the first one)
  if(is.data.frame(best_coords) || is.matrix(best_coords)) {
    threshold <- best_coords$threshold[1]
  } else {
    threshold <- best_coords[1]
  }
  
  # 3. Generate Predictions based on Best Threshold
  preds <- factor(ifelse(probs > threshold, "Yes", "No"), levels = c("No", "Yes"))
  
  # 4. Generate Confusion Matrix
  cm <- confusionMatrix(preds, truth, positive = "Yes", mode = "prec_recall")
  
  # 5. Extract Metrics
  metrics <- c(
    Kappa     = as.numeric(cm$overall['Kappa']),
    Sens      = as.numeric(cm$byClass['Sensitivity']),
    Spec      = as.numeric(cm$byClass['Specificity']),
    Precision = as.numeric(cm$byClass['Precision']),
    F1        = as.numeric(cm$byClass['F1']),
    AUC       = auc_val
  )
  
  # 6. OUTPUT: Print Confusion Matrix & Plot ROC
  cat(paste0("\n>>> MODEL: ", model_name, " <<<\n"))
  print(cm$table) 
  
  if(plot_roc) {
    plot(roc_obj, main = paste("ROC:", model_name), col = "blue", print.auc = TRUE)
  }
  
  return(list(metrics = metrics, threshold = threshold, preds = preds))
}

# 2. Execution -----------------------------------------------------------------

# Prepare the Plotting Area (2x2 Grid)
par(mfrow = c(2, 2)) 

# --- A. Physics Proxy Evaluation (Manual Offset Loop) ---
cat("\n>>> MODEL: Physics Proxy (Optimizing Offset) <<<\n")

# 1. Optimize Offset on TRAIN data (to avoid data leakage)
best_j <- -1
best_off <- 0

# Loop through offsets from -5 to +5
for(off in seq(-5, 5, 0.1)) {
  # Predict on TRAIN
  p_pred_train <- factor(ifelse(hk_train$DewPoint > (hk_train$Pred_GroundTemp + off), "Yes", "No"), levels=c("No","Yes"))
  
  # Calculate Simplified J Statistic (Sens + Spec - 1)
  tbl <- table(p_pred_train, hk_train$HNT_Label)
  # Safety check for empty classes
  if(all(dim(tbl) == c(2,2))) {
    sens <- tbl["Yes","Yes"] / sum(hk_train$HNT_Label == "Yes")
    spec <- tbl["No","No"]   / sum(hk_train$HNT_Label == "No")
    j <- sens + spec - 1
    
    if(j > best_j) {
      best_j <- j
      best_off <- off
    }
  }
}
cat(sprintf("Best Offset found (on Train): %.1f\n", best_off))

# 2. Apply Best Offset to TEST data
phys_pred <- factor(ifelse(hk_test$DewPoint > (hk_test$Pred_GroundTemp + best_off), "Yes", "No"), levels=c("No","Yes"))

# 3. Physics Metrics & Confusion Matrix
cm_phys <- confusionMatrix(phys_pred, hk_test$HNT_Label, positive = "Yes", mode = "prec_recall")
print(cm_phys$table)

# 4. Physics ROC (Binary)
# Note: For a fixed binary predictor, ROC is 3 points (0,0), (1-Spec, Sens), (1,1)
roc_phys <- roc(hk_test$HNT_Label, as.numeric(phys_pred == "Yes"), levels=c("No", "Yes"), quiet=TRUE)
plot(roc_phys, main = "ROC: Physics (Binary)", col = "red", print.auc = TRUE)

metrics_phys <- c(
  Kappa     = as.numeric(cm_phys$overall['Kappa']),
  Sens      = as.numeric(cm_phys$byClass['Sensitivity']),
  Spec      = as.numeric(cm_phys$byClass['Specificity']),
  Precision = as.numeric(cm_phys$byClass['Precision']),
  F1        = as.numeric(cm_phys$byClass['F1']),
  AUC       = as.numeric(roc_phys$auc)
)


# --- B. C5.0 Evaluation ---
probs_c50 <- predict(mod_c50, hk_test[, ai_features], type = "prob")[,"Yes"]
res_c50 <- evaluate_model("C5.0 (Hybrid)", hk_test$HNT_Label, probs_c50)

# --- C. Naive Bayes Evaluation ---
probs_nb <- predict(mod_nb, hk_test[, ai_features], type = "prob")[,"Yes"]
probs_nb[is.na(probs_nb)] <- 0 # Handle numeric stability issues
res_nb <- evaluate_model("Naive Bayes (Hybrid)", hk_test$HNT_Label, probs_nb)

# --- D. GLM Evaluation ---
probs_glm <- predict(mod_glm, hk_test[, ai_features], type = "prob")[,"Yes"]
res_glm <- evaluate_model("GLM (Hybrid)", hk_test$HNT_Label, probs_glm)

# Reset Plot Layout
par(mfrow = c(1, 1))

# 3. Final Consolidated Scorecard ----------------------------------------------

scorecard <- data.frame(
  Model = c("Physics Proxy", "C5.0", "Naive Bayes", "GLM"),
  rbind(
    metrics_phys,
    res_c50$metrics,
    res_nb$metrics,
    res_glm$metrics
  )
)

# Round for readability
scorecard[,-1] <- round(scorecard[,-1], 4)

cat("\n====================== FINAL HYBRID SCORECARD ======================\n")
print(scorecard)
cat("====================================================================\n")
