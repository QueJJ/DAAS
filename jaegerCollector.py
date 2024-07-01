import requests
import json
import numpy as np
import grpc
import csv

class JaegerCollector:
    def __init__(self, endpoint):
        self.endpoint = endpoint

    # Collect traces from Jaeger
    def collect(self, start_time, duration, limit, service, task_type, percentile, sub_file_name):
        request_data = {
            "start": int((start_time) * 1000000),
            "end": int((start_time + int(duration[:2]) * 1000) * 1000000),
            "operation": f"HTTP GET /{task_type}",
            "limit": limit,
            "service": service,
            "tags": '{"http.status_code":"200"}'
        }

        response = requests.get(self.endpoint, params=request_data)
        traces = json.loads(response.content)["data"]

        # Calculate per-service latency
        frontend_durations = []
        profile_durations = []
        reservation_durations = []
        search_durations = []
        geo_durations = []
        rate_durations = []
        
        for trace in traces:
            # duration: frontend, profile, reservation, search, geo, rate
            trace_duration_dict = {}
            for i in trace['spans']:
                operationName, duration = i['operationName'], i['duration']
                if operationName not in trace_duration_dict:
                    trace_duration_dict[operationName] = duration
                if (operationName in trace_duration_dict) and (trace_duration_dict[operationName] > duration):
                    trace_duration_dict[operationName] = duration

            if len(trace_duration_dict.keys()) == 6:
                tmp_search = 0
                for key, value in trace_duration_dict.items():
                    if 'profile.' in key:
                        profile_durations.append(value)
                    elif 'reservation.' in key:
                        reservation_durations.append(value)
                    elif 'search.' in key:
                        tmp_search = value
                    elif 'geo.' in key:
                        geo_durations.append(value)
                    elif 'rate.' in key:
                        rate_durations.append(value)

                search_durations.append(tmp_search - geo_durations[-1] - rate_durations[-1])
                frontend_durations.append(sum(trace_duration_dict.values()) - profile_durations[-1] - reservation_durations[-1] -\
                                           search_durations[-1] - geo_durations[-1] - rate_durations[-1])

        if len(frontend_durations) > 0:
            with open(sub_file_name, 'a', newline="") as f:
                for k in frontend_durations:
                    csv.writer(f).writerow([f"frontend", k])
                for k in profile_durations:
                    csv.writer(f).writerow([f"profile", k])
                for k in reservation_durations:
                    csv.writer(f).writerow([f"reservation", k])
                for k in search_durations:
                    csv.writer(f).writerow([f"search", k])
                for k in geo_durations:
                    csv.writer(f).writerow([f"geo", k])
                for k in rate_durations:
                    csv.writer(f).writerow([f"rate", k])

            L0_scv = np.array(frontend_durations)
            scv0 = np.var(L0_scv) / (np.mean(L0_scv) ** 2)
            L1_scv = np.array(profile_durations)
            scv1 = np.var(L1_scv) / (np.mean(L1_scv) ** 2)
            L2_scv = np.array(reservation_durations)
            scv2 = np.var(L2_scv) / (np.mean(L2_scv) ** 2)
            L3_scv = np.array(search_durations)
            scv3 = np.var(L3_scv) / (np.mean(L3_scv) ** 2)
            L4_scv = np.array(geo_durations)
            scv4 = np.var(L4_scv) / (np.mean(L4_scv) ** 2)
            L5_scv = np.array(rate_durations)
            scv5 = np.var(L5_scv) / (np.mean(L5_scv) ** 2)

            cp1 = np.array(frontend_durations) + np.array(profile_durations)
            cp2 = np.array(frontend_durations) + np.array(reservation_durations)
            cp3 = np.array(frontend_durations) + np.array(search_durations) + np.array(geo_durations)
            cp4 = np.array(frontend_durations) + np.array(search_durations) + np.array(rate_durations)

            cp = np.argmax([cp1, cp2, cp3, cp4], axis=0)
            counts = np.bincount(cp)
            maxCount = np.argmax(counts)

            # calculate percentile latency
            digit = 2
            if maxCount == 0:
                p0 = round(1 - scv0 / (scv0 + scv1) * 0.1, digit)
                p1 = round(1 - scv1 / (scv0 + scv1) * 0.1, digit)
                p2 = p1
                p3 = round(1 - min(max(scv4/(scv3+scv4), scv5/(scv3+scv5)), 1) * (0.1-(1-p0)), digit)
                p4 = round(1-p0 + 1-p3 + 0.9, digit)
                p5 = round(1-p0 + 1-p3 + 0.9, digit)
            elif maxCount == 1:
                p0 = round(1 - scv0 / (scv0 + scv2) * 0.1, digit)
                p2 = round(1 - scv2 / (scv0 + scv2) * 0.1, digit)
                p1 = p2
                p3 = round(1 - min(max(scv4/(scv3+scv4), scv5/(scv3+scv5)), 1) * (0.1-(1-p0)), digit)
                p4 = round(1-p0 + 1-p3 + 0.9, digit)
                p5 = round(1-p0 + 1-p3 + 0.9, digit)
            elif maxCount == 2:
                TmpList = [scv0 / (scv0 + scv3 + scv4), scv3 / (scv0 + scv3 + scv4), scv4 / (scv0 + scv3 + scv4)]
                TmpList = np.array(TmpList)
                p0 = round(1 - TmpList[0] * 0.1, digit)
                p3 = round(1 - TmpList[1] * 0.1, digit)
                p4 = round(1 - TmpList[2] * 0.1, digit)
                p1, p2, p5 = round(1-p0+0.9, digit), round(1-p0+0.9, digit), p4
            elif maxCount == 3:
                TmpList = [scv0 / (scv0 + scv3 + scv5), scv3 / (scv0 + scv3 + scv5), scv5 / (scv0 + scv3 + scv5)]
                TmpList = np.array(TmpList)
                p0 = round(1 - TmpList[0] * 0.1, digit)
                p3 = round(1 - TmpList[1] * 0.1, digit)
                p5 = round(1 - TmpList[2] * 0.1, digit)
                p1, p2, p4 = round(1-p0+0.9, digit), round(1-p0+0.9, digit), p5

                frontend_durations.sort()
                profile_durations.sort()
                reservation_durations.sort()
                search_durations.sort()
                geo_durations.sort()
                rate_durations.sort()

                frontend_percentile, profile_percentile, reservation_percentile = p0,p1,p2
                search_percentile, geo_percentile, rate_percentile = p3,p4,p5

                tail_frontend_idx = int(len(frontend_durations)*frontend_percentile)-1
                tail_profile_idx = int(len(profile_durations)*profile_percentile)-1
                tail_reservation_idx = int(len(reservation_durations)*reservation_percentile)-1
                tail_search_idx = int(len(search_durations)*search_percentile)-1
                tail_geo_idx = int(len(geo_durations)*geo_percentile)-1
                tail_rate_idx = int(len(rate_durations)*rate_percentile)-1

                tail_frontend_latency = frontend_durations[tail_frontend_idx]
                tail_profile_latency = profile_durations[tail_profile_idx]
                tail_reservation_latency = reservation_durations[tail_reservation_idx]
                tail_search_latency = search_durations[tail_search_idx]
                tail_geo_latency = geo_durations[tail_geo_idx]
                tail_rate_latency = rate_durations[tail_rate_idx]

                return [tail_frontend_latency, tail_profile_latency, tail_reservation_latency,
                tail_search_latency, tail_geo_latency, tail_rate_latency]
        else:
            print("[Jaeger] No traces found.")
            return None