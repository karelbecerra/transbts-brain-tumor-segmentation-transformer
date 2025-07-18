from utils.tools import average

trace_file = open("log/trace.csv", "a")  # append mode

def trace(epoch, metrics_t, metrics_v):
  row = str(epoch) \
          + ", " + str( metrics_t['loss'][-1]) \
          + ", " + str( metrics_t['dice1'][-1]) \
          + ", " + str( metrics_t['dice2'][-1]) \
          + ", " + str( metrics_t['dice3'][-1]) \
          + ", " + str( metrics_v['loss'][-1]) \
          + ", " + str( metrics_v['dice1'][-1]) \
          + ", " + str( metrics_v['dice2'][-1]) \
          + ", " + str( metrics_v['dice3'][-1]) 
  trace_file.write(row + "\n")
  trace_file.flush()

def close():
  trace_file.close()
