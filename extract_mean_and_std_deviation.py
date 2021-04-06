import yaml
import numpy


# List all names of the trace sets defined in "traces.yml" which should be analyzed
names = {"FPGA_PRESENT_RANDOMIZED_CLOCK": True, "FPGA_PRESENT_TI_MISALIGNED": True}


# Open "traces.yml" and load the parameter definitions of the trace sets
traceinfo = {}
with open("traces.yml","r") as f:
    ydata = f.read()
    traceinfo = yaml.safe_load(ydata)


# Extract means and standard deviations for all points in the trace sets
statistics = {}
for name in names:
    trdata = traceinfo[name]
    nrfiles = trdata["nrfiles"]
    trperfile = trdata["tracesinfile"]
    tracelen = trdata["tracelen"]
    dt = eval(trdata["struct"])
    means = []
    M2s = []
    for filenr in range(nrfiles):
        fname = trdata["path"] + "/Traces_{}.dat".format(filenr+1)
        with open(fname,"rb") as f:
            data = numpy.fromfile(f, dtype=dt, count=trperfile)
        m = numpy.mean(data["trace"],0)
        h = (data["trace"] - m) * (data["trace"] - m)
        m2 = numpy.sum(h,0)
        means.append(m)
        M2s.append(m2)
    merged_means = means[0]
    merged_stds = M2s[0]
    for x in range(1, nrfiles):
        delta = means[x] - merged_means
        merged_stds = merged_stds + M2s[x] + delta * delta * (x * trperfile)/(x+1) 
        merged_means = merged_means + delta/(x+1)
    merged_stds = numpy.sqrt(merged_stds / (nrfiles*trperfile))
    marr = []
    for m in merged_means:
        marr.append(float(m))
    sarr = []
    for s in merged_stds:
        sarr.append(float(s))
    statistics[name] = {}
    statistics[name]["mean"] = marr
    statistics[name]["std"] = sarr


# Dump the means and standard deviations into "tracestats.yml"
with open("tracestats.yml","w") as f:
    txt = yaml.dump(statistics, f)