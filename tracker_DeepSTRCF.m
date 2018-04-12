% Set the path to your trackers
path_to_your_trackers = 'E:/Database/vot-toolkit-master_HCF/tracker';

if isempty(path_to_your_trackers)
    error('Set the path to your trackers!');
end

tracker_label = 'DeepSTRCF';
tracker_command = generate_matlab_command('vot_wrapper(''DeepSTRCF'', ''DeepSTRCF_VOT_setting'')', {[path_to_your_trackers '/STRCF/']});
tracker_interpreter = 'matlab';
tracker_trax = false;