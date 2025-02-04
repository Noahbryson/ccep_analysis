import os
from pathlib import Path
from BCI2kReader import BCI2kReader as b2k
import matplotlib.pyplot as plt
import numpy as np
import warnings
import re
import glob
from scipy.stats import ttest_1samp
from scipy import signal as sig
from signal_procesing import highpass, notch, CRP
import pickle
from VERA_PyBrain.modules.VERA_PyBrain import PyBrain


class LoadCCEPs:
    def __init__(self, 
        mainDir: Path,
        filelist: list[Path] = [],
        rerefType:str="bipolar",
        highpass_fc: float = 1.0,
        triggerState: str | float = "DC04",
        showTrigger: bool = False
        ):
        self.identifiers = []
        self.dataStructs = []
        if len(filelist)<1: #can pass a subset filelist containing full paths to files of interest for a given run.
            filelist = glob.glob(f'{mainDir.parent}/{mainDir.name}/*.dat')
        for f in filelist:
            # coding should be session_day_run_freq_anode_cathod_trials_stimamp
            with b2k.BCI2kReader(f) as fp:
                fpath = Path(f)
                identif = fpath.name.replace(".dat", "").split("_")
                print(f"-----------------\n{'__'.join(identif)}")
                try:
                    res = CCEPrun(bci=fp, identif=identif, triggerState=triggerState, rerefType=rerefType,highpass_fc=highpass_fc,showTrigger=showTrigger)
                    amp = identif[7]
                    anode = identif[4].upper()
                    cathode = identif[5].upper()
                    self.dataStructs.append(res)
                    self.identifiers.append({'anode':anode,'cathode':cathode,'amp':amp})
                    # res.run_CRP(dir = f.parent)
                    # res.plot_trajectory("KL", endtime=500,rawPlot=True)
                    print("-----------------\n")
                except KeyError as err:
                    print(f"\n{err}")
                    print(f"\n{fpath.name} is invalid\n-----------------")
            # plt.show()


class CCEPrun:
    def __init__(
        self,
        bci: b2k.BCI2kReader,
        identif,
        t1: float = -20,
        t2: float = 1000,
        rerefType: str = "bipolar",
        highpass_fc: float = 1.0,
        triggerState: str | float = "DCO4",
        showTrigger: bool = False
    ):
        """
        __init__ _summary_

        Args:
            bci (b2k.BCI2kReader): _description_
            identif (_type_): _description_
            rerefType (str, optional): _description_. Defaults to 'bipolar'.
            highpass (float, optional): _description_. Defaults to 1.0.
        """
        self.session = identif[0]
        self.day = identif[1]
        self.run = identif[2]
        self.freq = identif[3]
        self.amp = identif[7]
        self.anode = identif[4].upper()
        self.cathode = identif[5].upper()
        self.n_trials = identif[6]
        self.parameters = bci.parameters
        self.fname = "_".join(identif)
        self.units = "uV"
        channels = bci.parameters["ChannelNames"]
        self.fs = bci.samplingrate
        self.states = self.__checkStates(bci.states)
        channelIndex = self.__parseChannelIDs(channels)
        signals = bci.signals[channelIndex]
        channels = [channels[i] for i in channelIndex]
        self.trajectories, self.data = self.__process(
            rerefType, signals, channels, highpass_fc
        )
        self.epoch_idx = self.__epoch_intervals(
            triggerState=triggerState, show=showTrigger
        )
        if len(self.epoch_idx) > 2.5 * int(self.n_trials) or len(self.epoch_idx) == 0:
            raise KeyError(f"Trigger Signal on {triggerState} is Invalid")
        elif len(self.epoch_idx) > int(self.n_trials):
            print(
                f"alarming number of trials detected\nexpected ~{self.n_trials} found {len(self.epoch_idx)}"
            )
            self.epoch_idx = self.__epoch_intervals(
                triggerState=triggerState, show=showTrigger
            )
        else:
            print(f"{len(self.epoch_idx)} trials indexed")
        self.ccep_time, self.ccep_data, self.epoch_shape = self.__epoch_data(t1, t2)
        print(f"epoch trials x samples: {self.epoch_shape}")
        self.__removeStimChannels(rerefType)
        self.__removeBadChannels()

    def __removeBadChannels(self):
        stdev = np.zeros(len(self.data))
        lineMag = np.zeros(len(self.data))
        postStimLocs = np.where(self.ccep_time >= 15)[0]
        sin60 = np.sin(self.ccep_time[postStimLocs]*2*np.pi*60)
        for idx,(key,epochs) in enumerate(self.ccep_data.items()):
            vals = epochs[:,postStimLocs]
            window = sig.get_window('hann',Nx=int(self.fs/2))
            line_sig = np.tile(sin60,reps=(len(vals),1))
            corr = sig.correlate(vals.T,line_sig.T,method='fft')
            corrLags = sig.correlation_lags(len(vals[0]),len(line_sig[0]))
            corr = sig.correlate(vals[0],line_sig[0])
            stdev[idx] = np.mean(np.std(vals,axis=1))
            f,psd = sig.welch(vals,self.fs,window=window,axis=1)
            lineMag[idx] = np.mean(psd[:,np.where(f==60)[0]])
        lineLoc = np.where(lineMag > np.median(lineMag)*1000)[0]
        stdevLoc = np.where(stdev > np.median(stdev)*10)[0]
        nbin = 2
        while True:
            hist,bins =np.histogram(lineLoc,bins=nbin)
            
        # plt.hist(lineMag,bins=10)
        # plt.show()
        return 0
        
    
    def __removeStimChannels(self, rerefType: str):
        pattern = r"\d+"
        anode_n = int(re.findall(pattern, self.anode)[0])
        cathode_n = int(re.findall(pattern, self.cathode)[0])
        match rerefType.lower():
            case "bipolar":
                checkkeys = [
                    f"{self.anode}-b-{cathode_n}",
                    f"{self.cathode}-b-{anode_n}",
                ]
            case "lapacian":
                checkkeys = [
                    self.anode,
                    self.cathode,
                    f"{self.anode}-b-{cathode_n}",
                    f"{self.cathode}-b-{anode_n}",
                ]
            case "common":
                checkkeys = [self.anode, self.cathode]
            case _:
                raise ValueError("Improper Selection of Re-referencing Method")
        for i in checkkeys:
            if i in self.ccep_data.keys():
                self.ccep_data.pop(i)
            if i in self.data.keys():
                self.data.pop(i)
            for j in self.trajectories:
                if i in self.trajectories[j]:
                    loc = self.trajectories[j].index(i)
                    self.trajectories[j].pop(loc)
        # return 0

    def __checkStates(self, states: dict):
        """
        __checkStates removes redundant states with only one value

        Args:
            states (dict): states from bci2kReader

        Returns:
            res (dict): pruned states
        """
        res = {i: j for i, j in states.items() if np.max(np.diff(j)) >= 1}
        for i, j in res.items():
            if isinstance(j, (list, np.ndarray, tuple)) and len(j) == 1:
                res[i] = j[0]
        return res

    def __epoch_data(self, t1: float = 15, t2: float = 1000):
        """_epoch_data extracts epoch intervals as self.epoch_idx and epochs data. returns a timeseries starting at t1 ms after stim onset and ending at t2 ms after stimulus onset

        Args:
            t1 (int, optional): _description_. Defaults to 15.
            t2 (int, optional): _description_. Defaults to 1000.
        Returns:
            _type_: _description_
        """
        epochs = {}
        startOffet = int(self.fs * t1 / 1000)  # convert ms to samples
        stopOffset = int(self.fs * t2 / 1000)  # convert ms to samples
        numSamps = stopOffset - startOffet + 1
        for channel, timeseries in self.data.items():
            temp = np.zeros(
                (len(self.epoch_idx), numSamps)
            )  # m x n where m is number of stim trials and n is number of samples in time
            time_array = np.linspace(t1, t2, numSamps)

            for i, interval in enumerate(self.epoch_idx):
                dat = timeseries[
                    startOffet + interval[0] : stopOffset + interval[0] + 1
                ]
                if len(timeseries) - (startOffet + interval[0]) >= numSamps:
                    temp[i, :] = dat
            temp = temp[:, ~np.all(temp[1:] == temp[:-1], axis=0)]
            epochs[channel] = temp

        return time_array, epochs, temp.shape

    def __epoch_intervals(self, triggerState: str = "DC04", show=False):
        """
        __epoch_intervals _summary_

        Args:
            triggerState (str, optional): state to target if a different analog trigger signal is used than normal. Defaults to 'DC04'.
            show (bool, optional): plot edge onset and offset detection for sanity. Defaults to False.

        Returns:
            epoch_ints (np.ndarray) : m x 2 array of integers mapping stimulation onset and offset. epochs[:,0] are onsets, epochs[:,1] are offsets


        """
        trigger = self.states[triggerState]
        trigger = trigger - np.mean(trigger)
        # trigger = np.round(trigger / np.max(trigger))
        trigger[np.where(trigger < 0)[0]] = 0
        # peaks,peakVals = sig.find_peaks(trigger)
        binTrig = trigger.astype(bool)
        testFlag = False
        if testFlag:
            fig = plt.figure("trigger")
            ax = plt.axes()
            ax = fig.add_axes(ax)
            ax.set_title(self.fname)
            ax.plot(trigger / np.max(trigger))
            ax.plot(binTrig)
            plt.show()
        locs = np.where(binTrig == True)[0]
        # epoch_ints = [i for idx, i in enumerate(locs[1:]) if i - locs[idx - 1] > 1]
        # epoch_ints.insert(0, locs[0])
        diffs = np.diff(locs)
        rising_edges = locs[np.where(diffs > 1)[0] + 1]
        falling_edges = locs[np.where(diffs > 1)[0]]
        rising_edges = np.insert(rising_edges, 0, locs[0])
        falling_edges = np.append(falling_edges, locs[-1])
        epoch_ints = np.zeros((rising_edges.size, 2))
        epoch_ints[:, 0] = rising_edges
        epoch_ints[:, 1] = falling_edges
        print(len(epoch_ints))
        epoch_ints = epoch_ints.astype(int)
        # show=True
        if show:
            fig = plt.figure("trigger")
            ax = plt.axes()
            ax = fig.add_axes(ax)
            # ax.plot(trigger)
            ax.plot(binTrig)
            # for i in epochs:
            ax.vlines(rising_edges, 0, 0.5, colors="r")
            ax.vlines(falling_edges, 0, 0.5, colors="g")
            peaks = sig.find_peaks(binTrig)
            ax.vlines(peaks[0], -0.5, 0, "y")
            ax.set_title(self.fname)
            plt.show()
        epoch_ints = [i for i in epoch_ints if np.diff(i) > 5]
        return np.array(epoch_ints)

    def __process(self, rerefType, signals, channels, highcut, notchOrder=2):
        trajectories = [re.findall("[a-zA-z]+", i)[0] for i in channels]
        trajectories = set(trajectories)
        commonAvg = np.mean(signals, axis=0)
        trajDict = {}
        output = {}
        for traj in trajectories:
            if rerefType.lower().find("bip") > -1:
                data = [v for k, v in zip(channels, signals) if k.find(traj) > -1]
                traj_chans = []
                for idx, vals in enumerate(data[0:-1]):
                    label = f"{traj}{idx+1}-b-{idx+2}"
                    temp = self._bipolarReference(data[idx + 1], vals)
                    if highcut > 0:
                        temp = highpass(temp, self.fs, highcut)
                    # temp = notch(temp, self.fs, 60, 30, notchOrder)
                    # temp = notch(temp, self.fs, 120, 30, notchOrder)

                    # temp = notch(temp,self.fs,60,30,1)
                    traj_chans.append(label)
                    output.update({label: temp})
                trajDict[traj] = traj_chans
            elif rerefType.lower().find("com") > -1:
                data = [v for k, v in zip(channels, signals) if k.find(traj) > -1]
                traj_chans = []
                for idx, vals in enumerate(data):
                    # label = f'{traj}_{idx+1}'
                    label = f"{traj}{idx+1}"
                    temp = vals - commonAvg
                    if highcut > 0:
                        temp = highpass(temp, self.fs, highcut)
                    temp = notch(temp, self.fs, 60, 30, notchOrder)
                    temp = notch(temp, self.fs, 120, 30, notchOrder)
                    # temp = notch(temp,self.fs,60,30,1)
                    traj_chans.append(label)
                    output.update({label: temp})
                    traj_chans.append(label)
                trajDict[traj] = traj_chans

            elif rerefType.lower().find("lapl") > -1:
                data = [v for k, v in zip(channels, signals) if k.find(traj) > -1]
                traj_chans = []
                for idx, vals in enumerate(data):
                    if idx == 0:
                        label = f"{traj}{idx+1}-b-{idx+2}"
                        temp = self._bipolarReference(data[idx + 1], vals)
                        if highcut > 0:
                            temp = highpass(temp, self.fs, highcut)
                        temp = notch(temp, self.fs, 60, 30, notchOrder)
                        temp = notch(temp, self.fs, 120, 30, notchOrder)
                        # temp = notch(temp,self.fs,60,30,1)
                        traj_chans.append(label)
                        traj_chans.append(label)
                        output.update({label: temp})
                    elif idx == len(data) - 1:
                        label = f"{traj}{idx}-b-{idx+1}"
                        temp = self._bipolarReference(vals, data[idx - 1])
                        # temp = notch(temp,self.fs,60,30,1)
                        if highcut > 0:
                            temp = highpass(temp, self.fs, highcut)
                        temp = notch(temp, self.fs, 60, 30, notchOrder)
                        temp = notch(temp, self.fs, 120, 30, notchOrder)
                        traj_chans.append(label)
                        traj_chans.append(label)
                        output.update({label: temp})
                    else:
                        label = f"{traj}{idx+1}-laplace"
                        temp = self._laplacianRereference(
                            vals, data[idx + 1], data[idx - 1]
                        )
                        if highcut > 0:
                            temp = highpass(temp, self.fs, highcut)
                        temp = notch(temp, self.fs, 60, 30, notchOrder)
                        temp = notch(temp, self.fs, 120, 30, notchOrder)
                        # temp = notch(temp,self.fs,60,30,1)
                        traj_chans.append(label)
                        traj_chans.append(label)
                        output.update({label: temp})
                trajDict[traj] = traj_chans

        trajDict = alphaSortDict(trajDict)
        return trajDict, output

    def _bipolarReference(self, a, b):
        return b - a

    def _laplacianRereference(
        self, data: np.ndarray, upstream: np.ndarray, downstream: np.ndarray
    ):
        x = data - np.mean((upstream, downstream), axis=0)
        return x

    def __parseChannelIDs(self, channels):
        EEG = [
            "FP1",
            "F3",
            "C3",
            "P3",
            "O1",
            "FP2",
            "F4",
            "C4",
            "P4",
            "O2",
            "F7",
            "T7",
            "P7",
            "F8",
            "T8",
            "P8",
            "F9",
            "F10",
            "FPZ",
            "FZ",
            "CZ",
            "PZ",
            "OZ",
        ]
        ECG = ["ECG1", "ECG2", "EKG1", "EKG2"]
        REF = ["REF1", "REF2"]
        agg = EEG + ECG + REF
        # res = []
        # for i,j in enumerate(channels):
        #       if not(j.upper().find('EMPTY')>-1 or j in agg):
        #             res.append(i)
        res = [
            i
            for i, j in enumerate(channels)
            if not (j.upper().find("EMPTY") > -1 or j in agg)
        ]
        return res

    def plot_trajectory(self, traj: str, endtime=200, rawPlot: bool = False):
        title = f"{'-'.join([self.anode,self.cathode])} {traj} {self.session} {self.day} R{self.run}"
        keys = self.trajectories[traj]
        cols = int(np.sqrt(len(keys)))
        rows = int(np.ceil(len(keys) / cols))
        fig, ax = plt.subplots(rows, cols, num=title, sharex=True, sharey=False)
        # fig, ax = plt.subplots(rows, cols, num=fig.number, sharex=True, sharey='row')
        locs = np.where(self.ccep_time <= endtime)[0]
        time = self.ccep_time[locs]
        pre_window = np.where(self.ccep_time <= 15)[0]
        pre_window_t = self.ccep_time[pre_window]
        for i, k in enumerate(keys):
            dat = self.ccep_data[k]
            a = ax.ravel()[i]
            if rawPlot:
                a.plot(time, dat[:, locs].T, color=(0, 0, 0), alpha=0.1)
            else:
                plot_range_on_curve(
                    time,
                    np.mean(dat[:, locs], axis=0),
                    np.std(dat[:, locs], axis=0),
                    a,
                    color=(0, 0, 0),
                )
            a.plot(time, np.mean(dat[:, locs], axis=0), color=(0, 0, 1), alpha=0.75)
            a.plot(
                pre_window_t,
                np.mean(dat[:, pre_window], axis=0),
                color=(1, 0, 0),
                alpha=1,
            )
            a.set_title(k)
        fig.suptitle(f"{self.fname}\n{self.epoch_shape[0]} Trials")
        return fig

    def run_CRP(
        self,
        load: bool = True,
        save: bool = False,
        dir: Path = None,
        prune: bool = False,
        postStimTime: float = 15,
    ):
        crp = CRP()
        parameters = {}
        projections = {}
        t_slice = np.where(self.ccep_time >= postStimTime)[0]
        time = self.ccep_time[t_slice]
        if not dir == None:
            file_loc = dir / "crp"
            os.makedirs(dir / "crp", exist_ok=True)
            fp = file_loc / f"{self.fname}_crp-params.pkl"
            if load and os.path.exists(fp):
                with open(fp, "rb") as f:
                    parameters = pickle.load(f)
            else:
                for i, j in self.ccep_data.items():
                    dat = j[:, t_slice].T
                    # parameters[i], projections[i] = crp.crp_method(dat.T,time,prune)
                    parameters[i], proj = crp.crp_method(dat, time, prune)
                    print(i)
                if save:
                    with open(fp, "wb") as f:
                        pickle.dump(parameters, f)
        return parameters, projections


class CCEP_divergent(LoadCCEPs):

    def __init__(self, targetChannel, brain: PyBrain, **kwargs):
        filelist = self.__getTargetFiles(fp = kwargs['mainDir'],targetChannel=targetChannel)
        super(CCEP_divergent, self).__init__(filelist=filelist,**kwargs)
        
    def __getTargetFiles(self,fp,targetChannel: str):
        targetChannel = targetChannel.upper()
        files = glob.glob(f'{fp}/*.dat')
        output = []
        for f in files:
            f = Path(f)
            fps = f.name.replace('.dat','').split('_')
            configs = ['-b-'.join([fps[4],fps[5]]),'-b-'.join([fps[5],fps[4]])]
            configs = [i.upper() for i in configs]
            if targetChannel in configs:
                output.append(fp/f)
        return output
    
    def plot_trajectories(self,endTime:float,stimAmp=6):
        datastruct = self.__parseAmplitude(stimAmp)
        for i in datastruct.trajectories:
            datastruct.plot_trajectory(i,endtime=endTime)
    def __parseAmplitude(self,stimAmp)->CCEPrun:
        for idx,i in enumerate(self.identifiers):
            if float(i['amp']) == stimAmp:
                return self.dataStructs[idx]
                
class CCEP_convergent(LoadCCEPs):
    def __init__(self, targetChannel, brain: PyBrain, **kwargs):
        super(CCEP_convergent, self).__init__(**kwargs)
        self.__parseData(targetChannel)
    
    def __parseData(self,targetChannel):
        
        return 0 


def plot_range_on_curve(t, curve, bounds, ax: plt.axes, color):
    upper = np.add(curve, bounds)
    lower = np.subtract(curve, bounds)
    # x = np.linspace(0,len(upper),len(upper))
    ax.fill_between(t, upper, lower, color=color, alpha=0.2, label="_")
    return ax


def alphaSortDict(a: dict) -> dict:
    sortkeys = sorted(a)
    output = {k: a[k] for k in sortkeys}
    return output


if __name__ == "__main__":
    dataPath = Path(
        r"/Users/nkb/Library/CloudStorage/Box-Box/Brunner Lab/DATA/SCAN_Mayo")
    subject = 'BJH041'
    gammaRange = [70,170]
    session = 'pre_ablation'
    brainType = "MNIbrain_destrieux"
    bp = dataPath/subject/'brain'/f'{brainType}.mat'
    maindir = dataPath/f'{subject}/stimulation_mapping/CCEPs/post-ablation'
    maindir = dataPath/f'{subject}/stimulation_mapping/CCEPs/pre-ablation'
    brain = PyBrain(bp)
    print(f'\nLoaded {brainType} for {subject} from {bp}\n')
    
    # subdir = "day19"
    targetChannel = 'KL9-b-KL10'
    results = CCEP_divergent(targetChannel=targetChannel,mainDir=maindir,brain=brain)
    results.plot_trajectories(500,stimAmp=3)
    plt.show()
    # results = CCEP_convergent(targetChannel=targetChannel,mainDir=maindir,brain=brain)
