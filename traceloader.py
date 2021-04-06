import yaml
import numpy

class TraceConfig():

    # Read the "traces.yml" and "tracestats.yml" files. Make sure that the "extract_mean_and_std_deviation.py" script has been executed after a new trace set is defined in the "traces.yml"
    def __init__(self):
        with open("traces.yml","r") as f:
            ydata = f.read()
            self.traceinfo = yaml.safe_load(ydata)
        with open("tracestats.yml","r") as f:
            ystat = f.read()
            self.statdata = yaml.safe_load(ystat)


    # Fetching and normalizing one trace file
    def get_normalized_from_file(self, tset, filenr):
        trdata = self.traceinfo[tset]
        nrfiles = trdata["nrfiles"]
        trperfile = trdata["tracesinfile"]
        tracelen = trdata["tracelen"]
        dt = eval(trdata["struct"])
        fname = trdata["path"] + "/Traces_{}.dat".format(filenr+1)
        with open(fname,"rb") as f:
            data = numpy.fromfile(f, dtype=dt, count=trperfile)
        m = self.statdata[tset]["mean"]
        s = self.statdata[tset]["std"]
        onlytraces = data["trace"]
        net_x = numpy.empty([trperfile,tracelen], dtype="float32")
        net_y = numpy.empty([trperfile, 2], dtype="u1")
        net_x = (onlytraces - m ) / s
        net_y[:,0] = data["group"]
        net_y[:,1] = 1 - data["group"]
        return net_x, net_y


    # Balancing the groups in the training and validation sets
    def balance(self, val_x, val_y, buf_x, buf_y):
        g = [0,0]
        g[0] = sum(val_y[:,0] == 1)
        g[1] = sum(val_y[:,1] == 1)
        ovindex = 0 if g[0] > g[1] else 1
        violations = (g[ovindex]-g[1-ovindex]) // 2
        valindex = len(val_x) - 1
        bufindex = 0
        bufsize = len(buf_y)
        for _ in range(violations):
            while val_y[valindex,ovindex] == 0:
                valindex -= 1
            while buf_y[bufindex,ovindex] == 1:
                bufindex += 1
                if bufindex >= bufsize:
                    print("best effort balancing, buf to small")
                    break
            val_x[valindex] = buf_x[bufindex]
            val_y[valindex,0] = buf_y[bufindex,0]
            val_y[valindex,1] = buf_y[bufindex,1]
            bufindex += 1
            valindex -=1


    # Return the number of points per trace defined in "traces.yml" for the given trace set
    def getnrpoints(self, tset):
        trdata = self.traceinfo[tset]
        return trdata["tracelen"]


    # Return the peak-to-peak distance in the traces defined in "traces.yml" for the given trace set
    def getpeakdistance(self, tset):
        trdata = self.traceinfo[tset]
        return trdata["peakdist"]


    # Preparing the required training and validation set with the given parameters
    def prep_traces(self, tset, nrtrain, nrval, balance = True):
        nrtraces = nrtrain + nrval
        print("Preprocessing of ({},{}) traces from {}...".format(nrtrain,nrval,tset))
        trdata = self.traceinfo[tset]
        nrfiles = trdata["nrfiles"]
        trperfile = trdata["tracesinfile"]   
        filelim = (nrtraces // trperfile) + 1
        assert filelim <= nrfiles
        for _ in range(4):
            if filelim + 1 <= nrfiles:
                filelim += 1
        nrload = filelim * trperfile
        net_x = numpy.empty([nrload, self.getnrpoints(tset)], dtype="float32")
        net_y = numpy.empty([nrload, 2], dtype="u1")
        for i in range(filelim):
            net_x[i*trperfile:(i+1)*trperfile], net_y[i*trperfile:(i+1)*trperfile] = self.get_normalized_from_file(tset, i)
        train_x = net_x[:nrtrain]
        train_y = net_y[:nrtrain]
        val_x = net_x[nrtrain:nrtraces]
        val_y = net_y[nrtrain:nrtraces]
        buf_x = net_x[nrtraces:]
        buf_y = net_y[nrtraces:]
        print("Before balancing:")
        g0 = sum(val_y[:,0] == 1)
        g1 = sum(val_y[:,1] == 1)
        print("#g0: {}, #g1: {}".format(g0,g1))
        print(" - size train: {} MB".format(train_x.nbytes /10**6))
        print(" - size val: {} MB".format(val_x.nbytes /10**6))
        if balance:
            self.balance(val_x,val_y,buf_x,buf_y)
            g0 = sum(val_y[:,0] == 1)
            g1 = sum(val_y[:,1] == 1)
            print("After balancing:")
            print("#g0: {}, #g1: {}".format(g0,g1))
            print(" - size train: {} MB".format(train_x.nbytes /10**6))
            print(" - size val: {} MB".format(val_x.nbytes /10**6))
        return train_x, train_y, val_x, val_y