
class VelocityEngine:
    """Velocity from supplied history only. If no real per-user history exists, caller must label it simulated/unavailable."""
    def __init__(self, window_minutes=10, high_count=5): self.window=window_minutes*60; self.high=high_count
    def score(self, timestamps: list[float], current_time: float):
        if not timestamps: return {"score":0.0,"count_in_window":0,"available":False,"note":"No historical per-user transactions available in this dataset."}
        cnt=sum(1 for t in timestamps if current_time-self.window <= t <= current_time)
        return {"score":min(1.0,cnt/self.high),"count_in_window":int(cnt),"available":True,"note":"Computed from supplied transaction history."}
    def simulated_score(self, count_in_window: int):
        return {"score":min(1.0,count_in_window/self.high),"count_in_window":int(count_in_window),"available":False,"simulated":True,"note":"SIMULATED velocity scenario - not observed user behavior."}
