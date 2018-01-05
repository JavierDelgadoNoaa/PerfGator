import os
import numpy as np
from collections import defaultdict
import gzip
import logging
import importlib
from ConfigParser import ConfigParser, NoSectionError, NoOptionError 
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distutils.util import strtobool
"""
Utilities to facilitate reading Tau output and generating figures.
 - More quick (and scriptable) way of generating plots
 - Ignore non-hotspots
 - Account for different roles of nodes/threads in parallel jobs 
   (e.g. I/O servers)

NOTES
- Nomenclature:
 * Code uses "subroutine" ("subr" for short), since "function" can have multiple
   meanings that lead to confusion (e.g. "mean"/"max" are functions)

-if num_calls is high, overhead is probably high too
"""

class NMMBConfig(object):
    """
    TODO : Move this to nwpy metcomp
    Encapsulate the options set in an NMM-B configuration file
    """
    def __init__(self, configFile):
        with open(configFile) as cfgFile:
            lines = cfgFile.readlines()
        for line in lines:
            if len(line) < 1: continue
            if line.startswith("inpes:"):
                self.num_pes_i = int(line.split()[1])
            elif line.startswith("jnpes:"):
                self.num_pes_j = int(line.split()[1])
            elif line.startswith("write_groups:"):
                self.num_write_groups = int(line.split()[1])
            elif line.startswith("write_tasks_per_group"):
                self.num_write_tasks_per_group = int(line.split(" ")[1])

class TauNodeContextThread(object):
    """
    Encapsulates the data for a given node, context, and thread
    """
    def __init__(self, node, context, thread, role=None):
        self.node = node
        self.context = context
        self.thread = thread
        # User-assigned role of this node/context/thread (e.g. compute task, I/O server, etc.)
        self.role = role
        self.subr_list = []
        self._total_time = None
    
    @property 
    def total_time(self):
        """Total time taken by the program. This is not necessarily set 
            initially since it may need to be estimated based on MAIN()"""
        if not self._total_time:
            # Attempt to calculate total_time based on known parameters of other subroutines 
            # that have already been processed
            for subr in self.subr_list:
                if subr.inclusive_percent > 10.:
                    self.total_time = subr.inclusive_time * (100. / subr.inclusive_percent)
                    break 
            if not self._total_time: raise UnknownParameterException("Total time is not known yet. Try using heuristics")
        return self._total_time

    @total_time.setter 
    def total_time(self, value):
        self._total_time = value

class UnknownParameterException(Exception):
    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)

#class NodeContextThreadItem(object):
class NCTCalltreeLeaf(object):
    """ 
    Encapsulates statistics for a particular subroutine call tree for this node,
    context, and thread (i.e. one line of Tau output)
    """
    def __init__(self, subr_name, exclusive_time, inclusive_time, inclusive_percent,
                 num_calls, nct, parents):
        self.subr_name = subr_name
        self.exclusive_time = exclusive_time
        self.inclusive_time = inclusive_time
        self.inclusive_percent = inclusive_percent 
        self.num_calls = num_calls 
        self.nct = nct
        self._exclusive_percent = None
        # This is simply a list of the subroutines in the call stack, since 
        # information about the call tree is not currently used
        self.parents = parents
   
    @property 
    def exclusive_percent(self):
        if not self._exclusive_percent: 
            self._exclusive_percent = self.exclusive_time / self.nct.total_time
            log.debug("jza Set exclusive percent for subr: {}".format(self.subr_name))
        return self._exclusive_percent

    def __eq__(self, other):
        
def _default_log(log2stdout=logging.INFO, log2file=None, name='el_default'):
    global _logger
    if _logger is None:
        _logger = logging.getLogger(name)
        _logger.setLevel(log2stdout)
        msg_str = '%(asctime)s::%(name)s::%(lineno)s::%(levelname)s - %(message)s'
        #msg_str = '%(asctime)s::%(funcName)s::%(filename)s:%(lineno)s::%(levelname)s - %(message)s'
        # TODO : Use this one if running in debug mode
        #msg_str = '%(levelname)s::%(asctime)s::%(funcName)s::%(filename)s:%(lineno)s - %(message)s'
        msg_str = '%(levelname)s :: %(filename)s:%(lineno)s - %(message)s'
        date_format = "%H:%M:%S"
        formatter = logging.Formatter(msg_str, datefmt=date_format)
        if log2file is not None:
            fh = logging.FileHandler('log.txt')
            fh.setLevel(log2file)
            fh.setFormatter(formatter)
            _logger.addHandler(fh)
        if log2stdout is not None:
            ch = logging.StreamHandler()
            ch.setLevel(log2stdout)
            ch.setFormatter(formatter)
            _logger.addHandler(ch)
    return _logger
_logger = None
        
def libstring_to_func(lib_and_func):
    """  
    Given a string containing the full module path to a function, including
    the function name, return the corresponding function (i.e. callback)
    @param lib_and_func The "fully qualified" function name.
    e.g. foo.bar.baz will return function baz from module foo.bar
    """
    lib_name = lib_and_func[0:lib_and_func.rindex(".")]
    func_name = lib_and_func[lib_and_func.rindex(".")+1:]
    lib = importlib.import_module(lib_name)
    func = getattr(lib, func_name)
    return func 

def dhms_to_sec(datestr):
    """ Convert date to seconds. `datestr' may be in
        msec with "," in the number or in DD:HH:MM:SS format"""
    #import pdb ; pdb.set_trace()
    if "," in datestr or not ":" in datestr: 
        try:
            return float(datestr.replace(",","")) / 1000.
        except:
            print 'jza333', datestr
            raise Exception()
    # [[D:[H:]]M:S format
    v = [float(s) for s in datestr.split(":")][::-1] # put in S:M:H:D
    for i in range(4 - len(v)): v.extend([0]) # add H:M:D if necessary
    return np.sum(np.array(v) * np.array([1., 60, 3600, 3600*24]))

def read_tau_output(filename="profile.txt", node2roleDict=None, 
                    subrsOfInterest=None, minExclusivePercent=1.0, 
                    minThresh={"exclusive_percent":1.0}, maxThresh={},
                    numSubrs=None, descending=True, sortMetric=None, 
                    rolesOfInterest=None):
    """
    This function reads a file containing TAU flat profile data (i.e. contents 
    of pprof) and creates an object-oriented interface to the profile data of
    interest.

    :param filename(str): Name of file containing profile data
    :param node2roleDict(dict):  dictionary mapping node number to role 
    :param subrsOfInterest(list): list of function names that matter. Any functions, 
        specified here will be included. i.e. this setting supersedes all others,
        so even if the thresholds are not satisfied, or if there are more than
        numSubrs, they will be included.
     TODO : Decide whether to use only name or callgraph
    :param minThresh(dict): dict that maps profile parameters to their minimum
        value needed to be included
        e.g. { "exclusive_percent": 1.0, "num_calls":100 } ignores all subroutines
        that took less than 1% of the time in the subroutine itself (excluding 
        children).
        DEFAULT: {"exclusive_percent": 1.0 } (NOTE: That if called from 
            gather_nmmb_tau_stats() a different default is used)
    :param: numSubrs: Number of subroutines in the profile to include. This 
                      includes number in ``subrsOfInterest'', if passed in.
    :param sortMetric: The metric to use when sorting subroutines to determine
                       which ones to include. Only used if numSubrs is passed in
        #TODO? default to paramOfInterest if unspecified?
    :param rolesOfInterest: If set, skip any nodes whose node2roleDict value
        does not match any item in this list
    :return nct_list,subr2nct: 
        nct_list is the list of TauNodeContextThread objects populated with all 
             NodeCtxThreadItem objects that satisfy passed in limitations.
        subr2nct is a dictionary that maps subroutines to the N/C/Ts that have 
            them in their subr_list (i.e. the N/C/T's for which the subroutine
            passed all thresholds and was either in subroutines_of_interest or
            one of the Top N resource consumers
    """
    ##
    # Variables
    ##
    # list of TauNodeContextThread objects
    profile = [] 
    # Dictionary mapping subroutines to the NCTs they belong to 
    subr2nct = defaultdict(list)
    # Do we need to re-run the filtering when all subroutines in a Node/Ctx/Thread
    # have been processed? This may be necessary, e.g. because the total_time
    # is not known until we encounter the ".Tau application" entry.
    redo_filtering = False
    temp_subr_list = []

    ##
    # Logic
    ##

    # Read all lines into memory
    if filename.endswith("gz"):
        lines = gzip.open(filename, "rb").readlines()
    else:
        lines = open(filename, 'r').readlines()

    # Determine whether we'll be getting additional subroutines beyond 
    # those specified via subrsOfInterest
    if subrsOfInterest and numSubrs:
        numSubsInListFound = 0
        if len(subrsOfInterest) > numSubrs:
                log.warn("The number of functions requested via ``subrsOfInterest''"
                         " [{0}] exceeds the number requested [{1}]. All functions"
                         " will be included.".format(len(subrsOfInterest), numSubrs))
        elif len(subrsOfInterest) == numSubrs:
            pass # Nothing to do or log
        else:
            log.debug("Requested to include more functions than specified via "
                      "``subrsOfInterest'', will include more, sorted by metric".format())
            assert sortMetric is not None
            num_additional_subrs = numSubrs - len(subrsOfInterest)
    else:
        numSubsInListFound = -999 # Unused if subrsOfInterest not specified
        num_additional_subrs = 0
    
    # Loop thru output
    skip_until_next_nct = False
    for line in lines:
        
        if line.startswith("NODE"):
            # In some cases, the filtering for the current NCT must be rerun
            # because parameters (e.g. total_time) are not known until later
            #import pdb ; pdb.set_trace()
            log.debug("Parsing new N/C/T line: {0}".format(line))
            if redo_filtering:
                idc_to_remove = []
                # Get list of indices of subroutines that don't pass filter
                # TODO : TEST
                for param,minValue in minThresh.iteritems():
                    log.debug("param,minvalue = {}{}".format(param, minValue))
                    for idx,subr in enumerate(temp_subr_list):
                        if getattr(subr, param) < minValue:
                            log.debug("removing subr: {}".format(subr.subr_name))
                            idc_to_remove.append(idx)
                for param,maxValue in maxThresh.iteritems():
                    log.debug("param,maxvalue = {}{}".format(param, maxValue))
                    for idx,subr in enumerate(temp_subr_list):
                        if getattr(subr,param) > maxValue:
                            log.debug("removing subr: {}".format(subr.subr_name))
                            idc_to_remove.append(idx)
                # Remove items that do not pass filter
                temp_subr_list = [temp_subr_list[i] for i in \
                      range(len(temp_subr_list)) if not i in idc_to_remove]
                # Update subr_list and subr2nct
                nct.subr_list.extend(temp_subr_list)
                ###for subr in temp_subr_list:
                    ###subr2nct[subr].append(nct)
                temp_subr_list = []
            # Point nct to a new TauNodeContextThread object
            #(node,context,thread) = [ int(x[1][0]) for x in [y.split(" ") \
            (node,context,thread) = [ int(x[1]) for x in [y.split(" ") \
                                     for y in line.replace(":","").split(";")] ] 
            log.debug("New node,context,thread = {0}, {1}, {2}"
                      .format(node, context, thread))
            role = node2roleDict[node] if node2roleDict is not None else None
            
            nct = TauNodeContextThread(node, context, thread, role)
            profile.append(nct)
            skip_until_next_nct = False
            numSubsInListFound = 0
            redo_filtering = False
            _total_exclusive_tally = 0

            if rolesOfInterest and node2roleDict[node] not in rolesOfInterest:
                log.debug("Skipping node {0} ({1})".format(node, node2roleDict[node]))
                skip_until_next_nct = True
            continue
        elif skip_until_next_nct:
            continue
        elif line.startswith("USER EVENTS"):
            log.debug("Encountered 'USER EVENTS line'. This program is currently "
                      "incompatible with this since it thinks it is a subroutine "
                      "line. Skipping until next N/C/T".format())
            skip_until_next_nct = True
        elif line.startswith("FUNCTION SUMMARY"):
            # Formatting can get a little funny after this
            skip_until_next_nct = True

        # Parse the line data
        #  10.7    24:09.197    24:09.197           1           0 1449197717 \
        #      .TAU application => MAIN_NEMS => MPI_Finalize()
        try:
            incl_perc = float(line[0:5])                                          
        except ValueError:
            # we are not in a subroutine info line
            continue
        
        tokenize = False
        for char in [5, 18, 31, 43, 55, 66]:
            if line[char] != " ":
                lineDat = line.split()
                tokenize = True
                if len(lineDat) < 6:
                    raise Exception("Unexpected line format\n{0}. Cowardly bailing"
                                .format(line))

        log.debug("Parsing line: {0}".format(line))
        if tokenize:
            excl_time = dhms_to_sec(lineDat[1])
            incl_time = dhms_to_sec(lineDat[2])
            num_calls = int(float(lineDat[4])) # only float converts exp string
            subr_name = "".join(lineDat[6:])

        else:
            excl_time = dhms_to_sec(line[7:19])
            incl_time = dhms_to_sec(line[20:31])                       
            num_calls = int(line[56:66])
            subr_name = line[67:].strip()

        
        # TODO check for duplicate entry func_name (e.g. one with callgraph one w/o - compare nmmb runs w and w/o callgraph 
        
        taskFunc = NodeContextThreadItem(subr_name, excl_time, incl_time, incl_perc, 
                                     num_calls, nct)
        # If not in subrsOfInterest, check if it meets all criteria
        # (if not, keep looping)
        if not ( subrsOfInterest and subr_name in subrsOfInterest ):
            # Optimization: If no additional subroutines needed beyond those in 
            # subrsOfInterest, just keep going
            if num_additional_subrs == 0:
                continue
            # Skip any entries that do not satisfy user-specified constraints
            # Note: May not know certain params (e.g. exclusive percent yet 
            # since that is derived. In these cases, postpone filtering 
            for param,thresh in minThresh.iteritems():
                try:
                    # Heuristic: if the parameter is "exclusive_percent", we can just use 
                    # exclusive_time, since it doesn't require a calculation
                    if param == "exclusive_percent":
                        log.debug("Using `exclusive_time' instead of `exclusive_percent'")
                        param = "exclusive_time"
                        param_value = getattr(taskFunc, param)
                        thresh = nct.total_time * (thresh / 100.)
                    else: 
                        param_value = getattr(taskFunc, param)
                    if param_value < thresh:
                        log.debug("Skipping {0} since {1}<{2}"
                                  .format(taskFunc.subr_name, param_value, thresh))
                        continue
                except UnknownParameterException:
                    log.debug("Parameter {0} is not known (yet). Processing of "
                             "{1} deferred".format(param, subr_name))
                    redo_filtering = True
                    temp_subr_list.append(taskFunc)
            for param,thresh in maxThresh.iteritems():
                try:
                    # Heuristic: if the parameter is "exclusive_percent", we can just use 
                    # exclusive_time, since it doesn't require a calculation
                    if param == "exclusive_percent":
                        log.debug("Using `exclusive_time' instead of `exclusive_percent'")
                        param = "exclusive_time"
                        param_value = getattr(taskFunc, param)
                        thresh = nct.total_time * (thresh / 100.)
                    else: 
                        param_value = getattr(taskFunc, param)
                    if param_value > thresh:
                        log.debug("Skipping {0} since {1}<{2}"
                                  .format(taskFunc.subr_name, param_value, thresh))
                        continue
                except UnknownParameterException:
                    log.debug("Parameter {0} is not known (yet). Processing of "
                             "{1} deferred".format(param, subr_name))
                    redo_filtering = True
                    temp_subr_list.append(taskFunc)
        else:
            numSubsInListFound += 1
       
        # In subrsOfInterest and/or passed all filters, so add to list
        nct.subr_list.append(taskFunc)
        ###subr2nct[subr_name].append(nct)

        # optimization: skip until next Node/Context/Thread if we're done looking
        if len(subrsOfInterest) == numSubsInListFound and num_additional_subrs==0:
            skip_until_next_nct = True
            numSubsInListFound = 0 # TODO : I already do  this up there. Can I delete?
        
        # Another optimization: If total elapsed time is above a threshld, skip to next node
        _total_exclusive_tally += excl_time
        try:
            #import pdb ; pdb.set_trace()
            perc = 100. * (_total_exclusive_tally / nct.total_time)
            #if perc > 99.5:
            if perc > 99.9: # and  len(subrsOfInterest) == numSubsInListFound:
                skip_until_next_nct = True
                numSubsInListFound = 0
        except UnknownParameterException:
            pass

    # Now, if a specific number of subroutines was requested (via ``numSubrs'')
    # and that number is less than number of elements in ``subrsOfInterest'',
    # sort the remaining routines by the desired metric and return the top N,
    # where N == numSubr - len(subrsOfInterest) 
    # NOTE: Gotta keep all routines in subrsOfInterest regardless of %
    for nct in profile:
       requested = [f for f in nct.subr_list if f.subr_name in subrsOfInterest]
       others = [f for f in nct.subr_list if not f.subr_name in subrsOfInterest]
       topN = sorted(others, key=lambda f: getattr(f, sortMetric), 
                     reverse=descending)[0:num_additional_subrs]
       #import pdb ; pdb.set_trace()
       nct.subr_list = requested
       nct.subr_list.extend(topN)
       # Map final list of subroutines to NCTs
       for subr in nct.subr_list:
            subr2nct[subr.subr_name].append(nct)
    assert [ len(nct.subr_list) == numSubrs for nct in profile ]
    return profile, subr2nct


def read_nmmb_tau_output(filename="profile.txt", configFile="configure_file_01",
                    subrsOfInterest=None,
                    minThresh={"exclusive_percent":1.0}, maxThresh={},
                    numSubrs=None, descending=True, sortMetric=None, 
                    rolesOfInterest=None):
    """
    Process TAU output from NMM-B. 
    This function will use the NMM-B configure_file to determine the process 
    distribution and use read_tau_output to generate a TauNodeContextThread for 
    each node.
    Then, different categories of nodes can be used for statistics gathering.
    e.g. mean of hot spots of compute tasks can be taken separte from i/o tasks

    :param subrsOfInterest(list): list of function names that matter. Any functions, 
        specified here will be included. i.e. this setting supersedes all others,
        so even if the thresholds are not satisfied, or if there are more than
        numSubrs, they will be included.
     TODO : Decide whether to use only name or callgraph
    :param minThresh(dict): dict that maps profile parameters to their minimum
        value needed to be included
        e.g. { "exclusive_percent": 1.0, "num_calls":100 } ignores all subroutines
        that took less than 1% of the time in the subroutine itself (excluding 
        children).
        DEFAULT: {"exclusive_percent": 1.0 }
    :param: numSubrs: Number of subroutines in the profile to include. This 
                      includes number in ``subrsOfInterest'', if passed in.
    :param sortMetric: The metric to use when sorting subroutines to determine
                       which ones to include. Only used if numSubrs is passed in
    :param rolesOfInterest: See documentation for read_tau_output()                
    :return nct_list, subr2values: The same output as read_tau_output()
    """
    def what_is_my_role(taskid):
        """
        Determine role of ``taskid'' in the NMM-B simulation.
        Will return ``compute'', ``write_master'', or ``write_slave''
        e.g. 0-95 = "compute", (96;100) = "write_master", 
            (97-99, 101-103) = "write_slave"
        """
        write_masters = range(num_compute, num_compute + num_io, 
                              cfg.num_write_tasks_per_group)
        write_slaves = list([ range(m+1 , m + cfg.num_write_tasks_per_group) \
                            for m in write_masters ])
        write_slaves = sum(write_slaves, []) # unpack list of lists

        if taskid < num_compute:
            return "compute"
        elif taskid in write_masters:
            return "write_master"
        elif taskid in write_slaves:
            return "write_slave"
        else:
            raise Exception("Could not determine role. This is a bug")
    #TODO? SKIP THREAD1
    cfg = NMMBConfig(configFile)
    num_compute = cfg.num_pes_i * cfg.num_pes_j
    num_io = (cfg.num_write_groups*cfg.num_write_tasks_per_group)
    #role_by_task = dict( (taskid,role) for i in range(num_compute + num_io))
    all_nodes = range(num_compute + num_io)
    node2role = dict( enumerate(what_is_my_role(n) for n in all_nodes) )
    return read_tau_output(filename, node2roleDict=node2role, 
                           subrsOfInterest=subrsOfInterest, 
                           minThresh=minThresh,maxThresh=maxThresh,
                           numSubrs=numSubrs, descending=descending, 
                           sortMetric=sortMetric, rolesOfInterest=rolesOfInterest)

 

def make_figures(exptProfiles, figconf):
    """
    :param exptProfiles: Dictionary : results[expt][role][operation][subroutine]
    """
    #compute_tasks = [t for t in nmmb_tasks if t.role == "compute" ]
    #io_tasks = [t for t in nmmb_tasks if t.role in ["write_slave", "write_master"]]
    #io_masters = [t for t in io_tasks if t.role == "write_master"]
    #io_slaves = [t for t in io_tasks if t.role == "write_slave"]
    
    
    if figconf["type"] == "stacked_bar":
        # Stacked bar chart
        plt.figure()
        startY = np.zeros(len(exptProfiles))
        xLocs = np.arange(len(exptProfiles))
        bar_width = float(figconf["barchart_width"])
        #import pdb ; pdb.set_trace() 
        color = []
        # Get set of all subroutines from all experiments
        expt_subrs = [ exptProfiles[x].keys() for x in exptProfiles.keys()]
        subrs = set([i for i in itertools.chain(*expt_subrs)])
        
        for subr in subrs:
            #subrs_of_interest[role]:
            # Get values for all experiments for the current role/op/subr combo
            # Extract value for each experiment. Note that not all experiments
            # will have same subroutines
            vals = []
            for i,exptKey in enumerate(exptProfiles.keys()):
                try:
                    vals.append(exptProfiles[exptKey][subr])
                except KeyError:
                    vals.append(0) #subr not in this experiment
            p = plt.bar(xLocs, vals, bar_width, bottom=startY)
            color.append(p[0])
            startY += vals
        plt.xticks(xLocs, exptProfiles.keys())
        plt.legend(color, subrs) # TODO use figconf setting
        plt.savefig(figconf["outfile"]) 
    else:
        raise Exception("Unknown figure type: {0}".format(figconf["type"]))

def get_figconf_minmax_thresh(figconf):
    """
    Given a dictionary with keys ``min_values'' and ``max_values'' containing
    strings in the format, ``paramA:N[, paramB:M], etc. return two dictionaries
    that map parameters to their min values
    """
    minthresh = {} ; maxthresh = {}
    if "min_values" in figconf:
        for minstr in figconf["min_values"].split(","):
            toks = minstr.split(":")
            assert len(toks) == 2
            minthresh[toks[0].strip()] = float(toks[1])
    #else:
    if "max_values" in figconf:
        for maxstr in figconf["max_values"].split(","):
            toks = maxstr.split(":")
            assert len(toks) == 2
            maxthresh[toks[0].strip()] = float(toks[1])
    return (minthresh, maxthresh)

def gather_nmmb_tau_stats(expt, figconf):
#operation, param):
    """
    Gather TAU statistics for an experiment
    :return subr2values, subr2numCases:
        subr2values is a dictionary mapping subroutine names to the value of 
        performing the operation specified in figconf to them. The number
        of subroutines in the dict is controlled by the settings in figconf.
        subr2numCases maps the subroutine names to the number of N/C/T's that
        contained this subroutine in its subr_list (i.e. it contains
        subroutines that were either specified via config or that happen
        to be the most resource intensive.
    """
    ## Settings
    global subrs_of_interest, num_subrs_to_include, subr_selection_sort_metric
    
    # TODO : Some subroutines have '()' some do not (in profile.txt). Also, some times there is a callgraph
    # other times there is not. Implement a user-friendly way of dealing with this.
    
    ## Variables
    subr2values = {} # map subroutines corresponding to this figconf to the value of performing ``operation'' on them
    subr2numCases = {} # map subroutines to the number of subroutines in which this was a hot spot
    operation = figconf["operation"]
    param = figconf["param_to_plot"]
    subrs_of_interest = figconf["subroutines_of_interest"]
    num_subr = figconf["num_subroutines"]
    sort_metric = figconf["sort_metric"]
    descending=strtobool(figconf["sort_descending"])
    expt_config = os.path.join(figconf["tau_outs_topdir"], expt, "configure_file_01")
    # Determine profile file path. TODO : Move this to subr...allow different file names
    prof_path = None
    dirname = os.path.join(os.getcwd(), figconf["tau_outs_topdir"], expt)
    for possible_file in ["profile.txt", "profile.txt.gz"]:
        if os.path.exists(os.path.join(dirname, possible_file)):
            prof_path = os.path.join(dirname, possible_file)
            break
    if not prof_path: 
        raise Exception("Did not find `profile.txt' or `profile.txt.gz' in {0}"
                       .format(dirname))

    
    ## Logic
    # Get list of NCTs whose subr_list satisfy the contstraints of
    # num_subrs_to_include, min/max values, etc.
    # TODO: DATABASE -> map canonical path and config obj to nct_list and subr2nct
    minthresh,maxthresh = get_figconf_minmax_thresh(figconf)
    nct_list,subr2nct = read_nmmb_tau_output(prof_path, expt_config, 
                        subrsOfInterest=subrs_of_interest, numSubrs=num_subr,
                        sortMetric=sort_metric, descending=descending,
                        rolesOfInterest=figconf["role_names"], 
                        minThresh=minthresh, maxThresh=maxthresh)
    #import pdb ; pdb.set_trace()
    # Put the value of ``paramToUse'' from all  NCTItem objects from
    # all NCTs of current role into a list
    # TODO : Confirm keys only contain final subroutines
    
    # Get the value of performing specified operation on subroutine
    # Note that not each subroutine will necessarily be in the ``subr_list''
    # of each NCT, due to load imbalance. 
    for subr in subr2nct.keys():
        subr_ncts = subr2nct[subr]
        # Get list of subroutines for each NCT that contains current subroutine
        nct_subr_names = [ [slist.subr_name for slist in nct.subr_list] \
                           for nct in subr_ncts]
        # Get index of current subroutine in each NCT's subroutine list
        # (for all NCT's that contain this current subroutine)
        #import pdb ; pdb.set_trace() 
        idc = [ subr_names.index(subr) for subr_names in nct_subr_names]
        # Get values of param of interest for all subrs in all applicable NCTs
        param_values = [getattr(nct.subr_list[idx], param) \
                        for idx,nct in zip(idc, subr_ncts)]
        # Issue callback to the operation of interest.
        opCall = libstring_to_func(operation)
        #import pdb ; pdb.set_trace()
        #role_profiles[roleId][op][param][subr] = opCall(param_values)
        subr2values[subr] = opCall(param_values)
        subr2numCases[subr] = len(idc)
    
    # Due to load imbalance, the number of subroutines may exceed the number
    # desired, so sort and purge again.
    # TODO : Have the option of sorting by numCases - since the having a 
    #       subroutine as a hot spot on more nodes is indicative of it
    #       being important
    #import pdb ; pdb.set_trace()
    # Requested subroutines may not all be there, so subtract from the ones that are
    requested = [f for f in subr2values.keys() if f in subrs_of_interest]
    num_additional_subrs = num_subr - len(requested) 
    if num_additional_subrs > 0:
        #others = [f for f in subr2values.keys() if f not in subrs_of_interest]
        subYval = [(sub,val) for (sub,val) in subr2values.iteritems() \
                  if sub not in subrs_of_interest]
        topN = sorted(subYval, key=lambda sv: sv[1],
                  reverse=descending)[0:num_additional_subrs]
        final_set = requested
        final_set.extend([sub for (sub,val) in topN])
        trimmed_subr2values = {}
        trimmed_subr2numCases = {}
        for subr in final_set:
            trimmed_subr2values[subr] = subr2values[subr]
            trimmed_subr2numCases[subr] = subr2numCases[subr]
        subr2values = trimmed_subr2values
        subr2numCases = trimmed_subr2numCases 
    
    return subr2values, subr2numCases

def get_figure_configs(conf):
    """
    Process the figure configuration sections of the ConfigParser ``conf'' and 
    return a dictionary that maps common subroutine configurations to figure 
    configurations. A ``subroutine configuration" is the combination of the 
    figure config sections, "subrs_of_interest", "num_subroutines", "sort_metric", 
    and "sort_descending". 
    """
    common_figconf_subr = defaultdict(list)

    sections = str2list(conf.get("DEFAULT", "figures"))
    for section in sections:
        cg = lambda item: conf.get(section, item)
        #import pdb ; pdb.set_trace()
        figconf = dict(conf.items(section))
        # create key for common_figconf_subr dict
        key = figconf["subroutines_of_interest"] + figconf["num_subroutines"] +\
              figconf["sort_metric"] + figconf["sort_descending"]
        # Now convert some of the items and add to list of figconfs with 
        # these subroutine settings
        figconf["num_subroutines"] = int(figconf["num_subroutines"])
        figconf["subroutines_of_interest"] = str2list(figconf["subroutines_of_interest"])
        try:
            figconf["role_names"] = str2list(figconf["role_names"])
        except NoOptionError:
            figconf["role_names"] = None # will use any role
        common_figconf_subr[key].append(figconf)

    log.debug("Found the following common figconfigs: {0}".format(common_figconf_subr))
    return common_figconf_subr 

def main():
    global log, expts, param_of_interest
    # TODO : Filter out thread idc != 0

    # NOTE: The numSubr passed to read_nmmb_tau_output() should be a bit higher
    # than we actually want, since different nodes may have different hotspots
    
    ##
    # Variables
    ##
    conf = ConfigParser()
    conf.read("config.conf") # TODO read from cmdline
    # Dict mapping experiments to their performance profiles (i.e. lists of NCTs)
    expt_profiles = {}
    #log = _default_log(logging.DEBUG)
    log = _default_log(logging.INFO)

    ##
    # Logic
    ##
    subrSets_to_figconfs = get_figure_configs(conf)
    # above call will map common subroutine settings in figure configs to 
    # the figure configs themselves. This way, the most time consuming
    # part of this program, which is read_tau_output() only needs to be called
    # once per common subroutine setting
    for _,figconfs in subrSets_to_figconfs.iteritems():
        for figconf in figconfs:
            fig = figconf["outfile"]
            expt_profiles[fig] = {}
            for expt in str2list(conf.get("DEFAULT", "experiments")):
                expt_profiles[fig][expt] = {}
                (roleId, op, param) = [ figconf[s] for s in \
                                        ["role_id", "operation", "param_to_plot"] ]
                p, num_cases = gather_nmmb_tau_stats(expt, figconf) # TODO use num_caes in make_figure
                # Populate expt_profiles
                '''
                if not roleId in expt_profiles[fig][expt]: 
                    expt_profiles[fig][expt][roleId] = {}
                if not op in expt_profiles[fig][expt][roleId]:  
                    expt_profiles[fig][expt][roleId][op] = {}
                if not param in expt_profiles[fig][expt][roleId][op]:  
                    expt_profiles[fig][expt][roleId][op][param] = {}

                expt_profiles[fig][expt][roleId][op][param] = p 
                '''
                expt_profiles[fig][expt] = p

            make_figures(expt_profiles[fig], figconf)
    
str2list = lambda  s : [ tok.strip() for tok in s.split(",")]

if __name__ == "__main__":
    main()
    #import tau
    #tau.run("main()")
    

