import os

case_id_col = {}
activity_col = {}
resource_col = {}
timestamp_col = {}
label_col = {}
pos_label = {}
neg_label = {}
dynamic_cat_cols = {}
static_cat_cols = {}
dynamic_num_cols = {}
static_num_cols = {}
filename = {}


logs_dir = "../labeled_logs_csv_processed"
#### Traffic fines settings ####

for formula in range(1,3):
    dataset = "traffic_fines_%s"%formula
    
    filename[dataset] = os.path.join(logs_dir, "traffic_fines_%s.csv"%formula)
    
    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    pos_label[dataset] = "deviant"
    neg_label[dataset] = "regular"

    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource", "lastSent", "notificationType", "dismissal"]
    static_cat_cols[dataset] = ["article", "vehicleClass"]
    dynamic_num_cols[dataset] = ["expense", "timesincelastevent", "timesincecasestart", "timesincemidnight", "event_nr", "month", "weekday", "hour", "open_cases"]
    static_num_cols[dataset] = ["amount", "points"]
        


#### BPIC2012 settings ####
bpic2012_dict = {"bpic2012": "BPIC2012.csv",
                 "bpic2012_accepted": "bpic2012_O_ACCEPTED-COMPLETE.csv",
                 "bpic2012_declined": "bpic2012_O_DECLINED-COMPLETE.csv"
                }

for dataset, fname in bpic2012_dict.items():

    filename[dataset] = os.path.join(logs_dir, fname)

    case_id_col[dataset] = "Case ID"
    activity_col[dataset] = "Activity"
    resource_col[dataset] = "Resource"
    timestamp_col[dataset] = "Complete Timestamp"
    label_col[dataset] = "label"
    neg_label[dataset] = "deviant"
    pos_label[dataset] = "regular"
    # features for classifier
    dynamic_cat_cols[dataset] = ["Activity", "Resource"]
    static_cat_cols[dataset] = []
    dynamic_num_cols[dataset] = ["hour", "weekday", "month", "timesincemidnight", "timesincelastevent", "timesincecasestart", "event_nr", "open_cases"]
    static_num_cols[dataset] = ['AMOUNT_REQ']
