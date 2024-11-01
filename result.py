import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib

results_dir = "results/cfs-test/"   

def get_results(): #read from response_times.csv
    data=pd.DataFrame()
    # read files sorted by cpu time
    for file in sorted(os.listdir(results_dir)):
        if file.endswith(".csv"):
            name=file.split(".")[0]
            # reading integer value from string
            cpu_time = int(''.join(filter(str.isdigit, name)))
            df = pd.read_csv(results_dir+file)
            # df.columns = [f"cpu_time={cpu_time}"]
            print(df)
            # change name of column to cpu_time+cpu_time
            data[f"cpu_time={cpu_time}"] = df["response_time"]
            # data = pd.concat([data,df],ignore_index=True,axis=1)
            # print(data)
    return data

if __name__ == "__main__":
    data = get_results()
    print(data)
    # create histogram of response times
    fig = plt.figure()
    data.hist(bins=100)
    # plt.close('all')
        
    # Create histogram of response times
    
    plt.xlabel("Response Time")
    plt.ylabel("Frequency")
    plt.title("Histogram of Response Times")
    plt.savefig("histogram.png")
    # show descriptive statistics
    print(data.describe())

