#!/usr/bin/python3
import argparse
import sys
import pandas as pd
from datetime import datetime
import json
import math


def parseArgs():
    parser = argparse.ArgumentParser(description='Generate CSV from log file.')
    parser.add_argument('logfile', type=str,
                        help='Log file for CSV')

    parser.add_argument('-m','--mode', choices=['ue','enb','epc','ping','iperfClient','iperfServer','cellSearch','vehicleLog','vehicleOut','channelSounder','gnuradioOfdm', 'pawprints_4G', 'pawprints_5G', 'nemo', 'mgen'],
    
                        help='Mode for parsing')
    parser.add_argument('-o','--output', type=str, default=sys.stdout,
                        help='output file for csv, default is ')
    
    parser.add_argument('--nemo-date', type=str, default=None,
                        help='Date at which the nemo log was recorded, in YYYY-MM-DD format')


    return parser.parse_args()


def fixPrefixes(value: str): # fixes the SI prefixes (e.g., k = 10^3)
    if 'm' in value: # milli
        value=value.replace("m","") # get it out
        valFloat=float(value)*0.001 # make it float and milli
        return str(valFloat)
    if 'k' in value: # kilo
        value=value.replace("k","") # get it out
        valFloat=float(value)*1000. # make it float and kilo
        return str(valFloat)
    if 'M' in value: # Mega
        value=value.replace("M","") # get it out
        valFloat=float(value)*1000000. # make it float and mega
        return str(valFloat)
    if '%' in value: # percentages
        value=value.replace("%","") # get it out
        valFloat=float(value)*0.01 # make it float and percent
        return str(valFloat)
    return(value)

def normalize_bandwidth_to_mbits(value_str, unit):
    val = float(value_str)

    if unit == "bits/sec":
        return val / 1e6
    elif unit == "Kbits/sec":
        print(f"encountered Kbits {val}")
        return val / 1e3
    elif unit == "Mbits/sec":
        return val
    else:
        return val




class LogParser:
    def __init__(self, logFile, outputFname, modeArgs):
        self.outputFname = outputFname
        self.logFile = logFile
        self.modeArgs = modeArgs
        with open(logFile) as f:
            self.raw = f.readlines()
        self.data = {}

    def parse_cellSearch(self):
        self.data = {
            "time":[],
            "Freq": [],
            "EARFCN":[],
            "PHYID": [],
            "PRB":[],
            "Ports":[],
            "PSS":[],
            "PSR":[]
        }
        try:
            for l in self.raw[1:]:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = (" ".join(l.split("]")[1:])).split(",")

                self.data["time"].append(ts)
                self.data["Freq"].append(dt[dt.index(" Found CELL MHz")+1])
                self.data["EARFCN"].append(dt[dt.index("  EARFCN")+1])
                self.data["PHYID"].append(dt[dt.index(" PHYID")+1])
                self.data["PRB"].append(dt[dt.index(" PRB")+1])
                self.data["Ports"].append(dt[dt.index("  ports")+1])
                self.data["PSS"].append(dt[dt.index(" PSS power dB")+1])
                self.data["PSR"].append(dt[dt.index("  PSR")+1])

        except Exception as e:
            print("Error parsing cell search log file: ", e)

    def parse_ue(self):
        self.data = {
            "time":[],
            "cc": [],
            "pci":[],
            "rsrp": [],
            "pl":[],
            "cfo":[],
            "mcsDl":[],
            "snr":[],
            "turbo":[],
            "brateDl":[],
            "blerDl":[],
            "ta_us":[],
            "mcsUl":[],
            "buff":[],
            "brateUl":[],
            "blerUl":[]
        }
        try:
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1], "%Y-%m-%d %H:%M:%S.%f")
                l = l.replace("|","")
                dt = l.split("]")[1][:-1].split(" ")

                dt = [i for i in dt if '' != i]
                # print(str(len(dt))+"..."+l)
                if len(dt) == 15 and dt[0] != "cc" and dt[0] != "Current":
                    self.data["time"].append(ts)
                    for index, j in enumerate(dt):
                        j = fixPrefixes(j)
                        self.data[list(self.data.keys())[index+1]].append(j)

        except Exception as e:
            print("Error parsing UE log file: ", e)

    def parse_enb(self):
        self.data = {
            "time":[],
            "rnti":[],
            "cqi":[],
            "ri":[],
            "mcsDl":[],
            "brateDl":[],
            "okDl":[],
            "nokDl":[],
            "(%)Dl":[],
            "snr":[],
            "phr":[],
            "mcsUl":[],
            "brateUl":[],
            "okUl":[],
            "nokUl":[],
            "(%)Ul":[],
            "bsr":[]
        }
        ## Multi User case??
        try:
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[1][:-1].split(" ")

                dt = [i for i in dt if '' != i]
                if len(dt) == 16 and dt[0] != "rnti":
                    self.data["time"].append(ts)
                    for index, j in enumerate(dt):
                        self.data[list(self.data.keys())[index+1]].append(j)

        except Exception as e:
            print("Error parsing ENB log file: ", e)

    def parse_epc(self):
        # There are too much diverse data. Therefore, I parsed it without any filtering
        self.data = {
            "time":[],
            "log":[]
        }
        try:
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[1].replace("\n","")
                if dt and not dt.isspace():
                    self.data["time"].append(ts)
                    self.data["log"].append(dt)

        except Exception as e:
            print("Error parsing EPC log file: ", e)

    def parse_ping(self):
        self.data = {
            "time":[],
            "size(byte)":[],
            "destination":[],
            "icmp_seq":[],
            "ttl":[],
            "pingtime":[]
        }

        prev_seq = None
        last_dest = None

        try:
            for l in self.raw:
                ts = datetime.strptime(
                    l.split("]")[0].split("[")[1],
                    "%Y-%m-%d %H:%M:%S.%f"
                )
                dt = l.split("]")[1].replace("\n","")

                if "icmp_seq" in dt:
                    parts = dt.split(" ")[1:-1]
                    parts.remove("bytes")
                    parts.remove("from")

                    size = parts[0]
                    dest = parts[1]
                    seq = int(parts[2].replace("icmp_seq=",""))
                    ttl = parts[3].replace("ttl=","")
                    ptime = parts[4].replace("time=","")

                    # Fill missing (lost) packets
                    if prev_seq is not None and seq > prev_seq + 1:
                        for missing_seq in range(prev_seq + 1, seq):
                            self.data["time"].append(ts)
                            self.data["size(byte)"].append("0")
                            self.data["destination"].append(last_dest)
                            self.data["icmp_seq"].append(missing_seq)
                            self.data["ttl"].append("0")
                            self.data["pingtime"].append("0")

                    # Store current packet
                    self.data["time"].append(ts)
                    self.data["size(byte)"].append(size)
                    self.data["destination"].append(dest)
                    self.data["icmp_seq"].append(seq)
                    self.data["ttl"].append(ttl)
                    self.data["pingtime"].append(ptime)

                    prev_seq = seq
                    last_dest = dest

            # Optional: normalize to float
            self.data["pingtime"] = [float(x) for x in self.data["pingtime"]]

        except Exception as e:
            print("Error parsing Ping log file: ", e)

    def parse_iperfServer(self):
        self.data = {
            "time":[],
            "ID":[],
            "Interval(sec)":[],
            "Transfer(MBytes)":[],
            "Bandwidth(MBits/sec)":[]
        }
        try:
            for l in self.raw:
                if "- - - - - - - -" in l:
                    break

                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[-1].replace("\n","").split(" ")
                if "sec" in dt or "MBytes" in dt:
                    dt = [i for i in dt if '' != i]
                    self.data["ID"].append(l.split("]")[1].split("[")[1].strip())
                    self.data["time"].append(ts)

                    # Detect bandwidth unit BEFORE removing it
                    bw_unit = None
                    if "Mbits/sec" in dt:
                        bw_unit = "Mbits/sec"
                    elif "Kbits/sec" in dt:
                        bw_unit = "Kbits/sec"
                    elif "bits/sec" in dt:
                        bw_unit = "bits/sec"


                    dt.remove("sec") if "sec" in dt else None
                    dt.remove("Mbits/sec") if "Mbits/sec" in dt else None
                    dt.remove("Kbits/sec") if "Kbits/sec" in dt else None
                    dt.remove("bits/sec") if "bits/sec" in dt else None
                    # Something strange with the space character here. It doesn't remove the match. TODO
                    dt.remove("MBytes") if "MBytes" in dt else None
                    dt.remove("MBytes") if "MBytes" in dt else None
                    dt.remove("KBytes") if "KBytes" in dt else None
                    dt.remove("KBytes") if "KBytes" in dt else None
                    dt.remove("Bytes") if "Bytes" in dt else None
                    dt.remove("Bytes") if "Bytes" in dt else None

                    if len(dt) >= 7:
                        self.data["Interval(sec)"].append(dt[0])
                        self.data["Transfer(MBytes)"].append(dt[2])

                        bw_mbits = normalize_bandwidth_to_mbits(dt[3], bw_unit)
                        self.data["Bandwidth(MBits/sec)"].append(bw_mbits)
                        continue

                    for index, j in enumerate(dt):
                        self.data[list(self.data.keys())[index+2]].append(j)
        except Exception as e:
            print("Error parsing iperf log file: ", e)

    def parse_mgen(self):
        self.data = {
            "time":[],
            "Interval(sec)":[],
            "Latency(sec)":[],
            "Bandwidth(MBits/sec)":[]
        }
        try:
            for l in self.raw[2:]:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[-1].replace("\n","").split(" ")
                if "REPORT" in dt:
                    self.data["time"].append(ts)
                    dt = [i.split(">")[-1] for i in dt if '' != i]
                    #self.data["ID"].append(dt[4])
                    for index, j in enumerate(dt):
                        if index == 6:
                            self.data["Interval(sec)"].append(j)
                        elif index == 7:
                            self.data["Bandwidth(MBits/sec)"].append(float(j) / 1000)
                        elif index == 11:
                            self.data["Latency(sec)"].append(j)

        except Exception as e:
            print("Error parsing mgen log file: ", e)


    def parse_iperfClient(self):
        self.data = {
            "time":[],
            "ID":[],
            "Interval(sec)":[],
            "Transfer(MBytes)":[],
            "Bandwidth(MBits/sec)":[],
            "Retr":[],
            "Cwnd(KBytes)":[]
        }
        try:
            for l in self.raw:

                if "- - - - - - - -" in l:
                    break

                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[-1].replace("\n","").split(" ")

                if "sec" in dt or "MBytes" in dt or "KBytes" in dt :
                    dt = [i for i in dt if '' != i]
                    self.data["ID"].append(l.split("]")[1].split("[")[1].strip())
                    self.data["time"].append(ts)

                    dt.remove("sec") if "sec" in dt else None
                    dt.remove("Mbits/sec") if "Mbits/sec" in dt else None
                    dt.remove("Kbits/sec") if "Kbits/sec" in dt else None
                    dt.remove("bits/sec") if "bits/sec" in dt else None
                    # Something strange with the space character here. It doesn't remove the match. TODO
                    dt.remove("MBytes") if "MBytes" in dt else None
                    dt.remove("MBytes") if "MBytes" in dt else None
                    dt.remove("KBytes") if "KBytes" in dt else None
                    dt.remove("KBytes") if "KBytes" in dt else None
                    dt.remove("Bytes") if "Bytes" in dt else None
                    dt.remove("Bytes") if "Bytes" in dt else None

                    for index, j in enumerate(dt):
                        self.data[list(self.data.keys())[index+2]].append(j)
        except Exception as e:
            print("Error parsing iperf log file: ", e)

    def parse_vehicleLog(self):
        self.data = {
            "time":[],
            "log":[]
        }
        try:
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[1].replace("\n","")
                if dt and not dt.isspace():
                    self.data["time"].append(ts)
                    self.data["log"].append(dt)

        except Exception as e:
            print("Error parsing Vehicle log file: ", e)

    def parse_vehicleOut(self):
        # this is the order both for predetermined trajectory and GPS_Logger vehicle logging
        # if you change one of them, please also change the other one
        self.data = {
            "num":[],
            "Longitude":[],
            "Latitude":[],
            "Altitude":[],
            "Pitch":[],
            "Yaw":[],
            "Roll":[],
            "VelocityX":[],
            "VelocityY":[],
            "VelocityZ":[],
            "BatteryVolts":[],
            "time":[],
            "GPSFix":[],
            "NumberOfSatellites":[]
        }
        try:
            for l in self.raw:
                l1=l.replace('"(',"") # fixing the grouping for attitude and velocities
                l1=l1.replace(')"',"") # fixing the grouping for attitude and velocities
                dt = l1.replace("\n","").split(",")
                for index, j in enumerate(dt):
                    self.data[list(self.data.keys())[index]].append(j)
                #
                # The following code seems to take care of some error or exception, but Anil doesn't recall
                # exactly what - it's quite possible that it was taken care of already and it's not needed
                # If it turns out that it's needed, please specify the problem and uncomment it
                #
                # t_ind = 5+6 # I'm guessing the index of the time field.
                # if len(dt) >= 8+6: # I'm not sure what 8 is :-), possibly  have a GPS fix?
                #    t_ind = 6+6
                #for index, j in enumerate(dt):
                #    if t_ind == 5+6:
                #        self.data[list(self.data.keys())[index]].append(datetime.strptime(j, "%Y-%m-%d %H:%M:%S.%f"))
                #    else:
                #        self.data[list(self.data.keys())[index]].append(j)

        except Exception as e:
            print("Error parsing Vehicle Out log file: ", e)

    def parse_channelSounder(self):
        self.data = {
            "time":[],
            "Measurement No":[],
            "Power in dB":[]
        }
        try:
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[-1].replace("\n","").split(" ")
                self.data["time"].append(ts)
                dt = [i for i in dt if '' != i]
                if len(dt) != 2:
                    continue

                for index, j in enumerate(dt):
                    self.data[list(self.data.keys())[index+1]].append(j)
        except Exception as e:
            print("Error parsing channel sounder log file: ", e)
        
    def parse_gnuradioOfdm(self):
        self.data = {
            "time":[],
            "Offset":[],
            "Source":[],
            "Key":[],
            "Value":[]
        }
        try:
            parseInd = 0
            for l in self.raw:
                ts = datetime.strptime(l.split("]")[0].split("[")[1],"%Y-%m-%d %H:%M:%S.%f")
                dt = l.split("]")[-1].replace("\n","")
                if "Tag Debug: Rx Bytes with SNR" in dt or "Input Stream:" in dt:
                    if parseInd != 2:
                        parseInd += 1
                        continue

                if parseInd == 2:
                    dt = dt.split(" ")
                    dt = [i for i in dt if '' != i]
                    self.data["time"].append(ts)

                    dt.remove("Offset:") if "Offset:" in dt else None
                    dt.remove("Source:") if "Source:" in dt else None
                    dt.remove("Key:") if "Key:" in dt else None
                    dt.remove("Value:") if "Value:" in dt else None

                    for index, j in enumerate(dt):
                        self.data[list(self.data.keys())[index+1]].append(j)
                    parseInd = 0
        except Exception as e:
            print("Error parsing Gnuradio OFDM log file: ", e)


    def parse_pawprints_4G(self):
        self.data = []
        try:
            for log_row_string in self.raw:
                log_row = json.loads(log_row_string)
                for cell in log_row["cells"]:
                    pd_cell = cell.copy()
                    # Ideally, the below operations should be performed upstream at the Android App.
                    pd_cell["phone_abs_time"] = log_row["abs_time"]
                    pd_cell["phone_time_readable"] = self._epoch_ms_to_readable(log_row["abs_time"])

                    # Indicate if this cell is the one to which the Android is connected
                    pd_cell["is_connected"] = 0
                    if int(cell["pci"]) == log_row["connected_pci"]:
                        pd_cell["is_connected"] = 1 

                    if "companion_abs_time" in log_row:
                        pd_cell["companion_abs_time"] = log_row["companion_abs_time"]
                        pd_cell["companion_time_readable"] = self._epoch_ms_to_readable(log_row["companion_abs_time"])

                    self.data.append(pd_cell)
                    
        except Exception as e:
            print("Error generating PawPrints CSV: ", e)

    def parse_pawprints_5G(self):
        self.data = []
        try:
            for log_row_string in self.raw:
                log_row = json.loads(log_row_string)
            
                if "nr_signal_strength" in log_row and len(log_row['nr_signal_strength'].keys()) > 1:
                    pd_cell = log_row["nr_signal_strength"].copy()
                    pd_cell["phone_abs_time"] = log_row["abs_time"]
                    pd_cell["phone_time_readable"] = self._epoch_ms_to_readable(log_row["abs_time"])

                    if "companion_abs_time" in log_row:
                        pd_cell["companion_abs_time"] = log_row["companion_abs_time"]
                        pd_cell["companion_time_readable"] = self._epoch_ms_to_readable(log_row["companion_abs_time"])

                    self.data.append(pd_cell)
                    
        except Exception as e:
            print("Error generating PawPrints CSV: ", e)


    def parse_pawprints(self):
        self.data = {"nr_signal_strength": None, "cell_info": None}
        self.data = {}
        
        try:
            for log_row_string in self.raw:
                log_row = json.loads(log_row_string)
            
                if "nr_signal_strength" in log_row and len(log_row['nr_signal_strength'].keys()) > 1:
                    pd_cell = log_row["nr_signal_strength"].copy()
                    pd_cell["phone_abs_time"] = log_row["abs_time"]
                    pd_cell["rel_time"] = log_row["rel_time"]
                    pd_cell["companion_abs_time"] = log_row["companion_abs_time"]
                    self.data["nr_signal_strength"] = pd.concat([self.data["nr_signal_strength"], pd.DataFrame(pd_cell, index =[0])], ignore_index = True)

                for cell in log_row["cells"]:
                    pd_cell = cell.copy()
                    pd_cell["phone_abs_time"] = log_row["abs_time"]
                    pd_cell["rel_time"] = log_row["rel_time"]    
                    pd_cell["is_connected"] = 0
                    if int(cell["pci"]) == log_row["connected_pci"]:
                        pd_cell["is_connected"] = 1 

                    if "companion_abs_time" in log_row:
                        pd_cell["companion_abs_time"] = log_row["companion_abs_time"]

                    # Indicates whether the current cell is the one that the phone is connected to
                    pd_cell["is_connected"] = 0
                    if cell["pci"] == log_row["connected_pci"]:
                        pd_cell["is_connected"] = 1

                    self.data.append(pd_cell)
                else:
                    print("Unsupported PawPrints log type.")
                    pass

                    self.data["cell_info"] = pd.concat([self.data["cell_info"], pd.DataFrame(pd_cell, index = [0])], ignore_index = True)
                    
        except Exception as e:
            print("Error generating PawPrints CSV: ", e)

    
    def _process_nemo_radio_log(self, nemo_raw_df, processed_nemo_json, cell_id_col, nemo_log_date):
        # Loop over PCIs. Split PCIs and append KPI of each unique PCI to the corresponding field of the JSON.
        # This assumes that number of PCI entries = number of KPI entries, per row, in raw nemo dataframe.
        row_type = "downlink"
        for index, row in nemo_raw_df.iterrows():
            # Dont process a row with an invalid time string
            if not self._validate_time_format(row["Time"], "%H:%M:%S.%f"):
                continue

            # Dont process a row with Time as NaN.  
            if not self._is_series_NonNaN(row["Time"]):
                continue
            
            # Dont process a row with all columns as NaN.  
            if not self._is_series_NonNaN(row.drop("Time")):
                continue

            pcis_str = row[cell_id_col]
            if not isinstance(pcis_str, str) and math.isnan(pcis_str):
                # PCI is NaN for a row with Uplink info. Dont encapsulate pcis in an array.
                row_type = "uplink"
                num_exploded_rows = 1
            elif isinstance(pcis_str, (int, float)):
                row_type = "downlink"
                pci_arr = [pcis_str]
                num_exploded_rows = len(pci_arr)
            elif isinstance(pcis_str, str):
                row_type = "downlink"
                pci_arr = pcis_str.split(",")
                num_exploded_rows = len(pci_arr)
        
            row["Time"] = nemo_log_date + " " + row["Time"]
            print(f"log line num:{index}")
            for kpi_name in processed_nemo_json:
                # If the KPI is not time, split the string to array, validate with pci count, and store in json. 
                if "time" not in kpi_name.lower():
                    kpi_arr = row[kpi_name]
                    if row_type == "uplink":
                        kpi_arr = [row[kpi_name]]
                    elif isinstance(kpi_arr, (int, float)): 
                        kpi_arr = [kpi_arr for i in range(num_exploded_rows)]
                    else:
                        kpi_arr = kpi_arr.split(",")
                        if len(kpi_arr) != len(pci_arr):
                            raise Exception("Nemo parsing error. KPI count != PCI cell count.")

                    for kpi_val in kpi_arr:
                        processed_nemo_json[kpi_name].append(kpi_val)
                else:
                    processed_nemo_json[kpi_name].extend([row[kpi_name] for i in range(num_exploded_rows)])

    
    def _process_nemo_non_cellular_logs(self, nemo_raw_df, processed_nemo_json, nemo_log_date):
        for index, row in nemo_raw_df.iterrows():
        
            # Dont process a row with an invalid time string
            if not self._validate_time_format(row["Time"], "%H:%M:%S.%f"):
                continue

            row["Time"] = nemo_log_date + " " + row["Time"]

            # Dont process a row with Time as NaN.  
            if self._is_variable_NaN(row["Time"]):
                continue
            
            # Dont process a row with all columns as NaN.  
            if not self._is_series_NonNaN(row.drop("Time")):
                continue

            for kpi_name in processed_nemo_json:
                processed_nemo_json[kpi_name].append(row[kpi_name])

    
    def _format_nemo_gps_times(self, str_time):
        epoch_time =  datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f').timestamp()*1000.0 # in milliseconds
        return epoch_time

    def parse_nemo(self):
        LTE_CELL_ID_FIELD = "Physical layer identity (LTE detected)"
        NR_CELL_ID_FIELD = "Physical cell identity (NR SpCell)"
        try:
            nemo_raw_df = pd.read_csv(self.logFile)
            cell_id_col = LTE_CELL_ID_FIELD if LTE_CELL_ID_FIELD in nemo_raw_df.columns else NR_CELL_ID_FIELD if NR_CELL_ID_FIELD in nemo_raw_df.columns else None 
            
            nemo_raw_df.dropna(axis=1, how="all") # Supposed to drop NaN rows but it does not. I drop it manually below.
            processed_nemo_json = {}

            # Initialize processed json with fields as the columns of the raw nemo dataframe
            for col in nemo_raw_df.columns:
                # Remove columns with all NaN values
                if self._is_series_NonNaN(nemo_raw_df[col]):
                    processed_nemo_json[col] = []

            if cell_id_col is not None:
                self._process_nemo_radio_log(nemo_raw_df, processed_nemo_json, cell_id_col, self.modeArgs["nemo_date"])
            else:
                self._process_nemo_non_cellular_logs(nemo_raw_df, processed_nemo_json, self.modeArgs["nemo_date"])

            processed_nemo_df = pd.DataFrame(processed_nemo_json)
            processed_nemo_df["nemo_abs_time"] = processed_nemo_df["Time"].apply(self._format_nemo_gps_times)
            self.data = processed_nemo_df.to_dict()

        except Exception as e:
            print("Error generating Nemo CSV: ", e)

    '''
    The below helper / library functions are meant to be re-used across parser functions
    '''

    def _epoch_ms_to_readable(self, epoch_time_ms):
        date_obj = datetime.fromtimestamp(epoch_time_ms/1000.0)
        return date_obj.strftime('%Y-%m-%d %H:%M:%S.%f')

    def _is_series_NonNaN(self, pd_series):
        term_wise_nans = pd.isna(pd_series)
        if isinstance(term_wise_nans, bool):
            return term_wise_nans is False
        else:
            return False in term_wise_nans.unique()

    def _validate_time_format(self, time_str, time_format):
        try:
            datetime.strptime(time_str, time_format)
            return True
        except Exception as e:
            return False
        
    def _is_variable_NaN(self, var):
        return isinstance(var, (float, int)) and math.isnan(var)


    def exportCsv(self):
        try:
            csvDf = pd.DataFrame.from_dict(self.data)
            csvFName = self.outputFname if ".csv" in self.outputFname else self.outputFname + ".csv"
            csvDf.to_csv(csvFName, index=False)
            print('Saved ' + str(csvDf.shape[0]) + ' lines of data in ' + csvFName)
            print('Available fields:')
            for col in csvDf.columns:
                print(col)
        except Exception as e:
            print("Error generating CSV: ", e)


def create_mode_args(args):
    mode_args = {}
    if args.nemo_date:
        mode_args["nemo_date"] = args.nemo_date

    return mode_args

def main():
    args = parseArgs()
    mode_args = create_mode_args(args)
    parser =  LogParser(args.logfile, args.output, mode_args)
    getattr(parser, "parse_" + args.mode)()
    parser.exportCsv()


if __name__ == '__main__':
    main()
