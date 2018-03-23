function results = run_DeepSTRCF(seq, res_path, bSaveImage)

% Feature specific parameters
hog_params.cell_size = 4;
hog_params.compressed_dim = 10;
hog_params.nDim = 31;

grayscale_params.colorspace='gray';
grayscale_params.cell_size = 4;

cn_params.tablename = 'CNnorm';
cn_params.useForGray = false;
cn_params.cell_size = 4;
cn_params.nDim = 10;

cnn_params.nn_name = 'imagenet-vgg-m-2048.mat'; % Name of the network
cnn_params.output_layer = [10];           % Which layers to use
cnn_params.downsample_factor = [1];           % How much to downsample each output layer
cnn_params.input_size_mode = 'adaptive';        % How to choose the sample size
cnn_params.input_size_scale = 1;                % Extra scale factor of the input samples to the network (1 is no scaling)

% Which features to include
params.t_features = {
    struct('getFeature',@get_cnn_layers, 'fparams',cnn_params),...
    struct('getFeature',@get_colorspace, 'fparams',grayscale_params),...    
    struct('getFeature',@get_fhog,'fparams',hog_params),...
    struct('getFeature',@get_table_feature, 'fparams',cn_params),...
};
    
% Global feature parameters
params.t_global.cell_size = 4;                  % Feature cell size

% Image sample parameters
params.search_area_shape = 'square';    % The shape of the samples
params.search_area_scale = 4.5;         % The scaling of the target size to get the search area
params.min_image_sample_size = 200^2;   % Minimum area of image samples
params.max_image_sample_size = 250^2;   % Maximum area of image samples

% Spatial regularization window_parameters
params.feature_downsample_ratio = [4, 14]; %  Feature downsample ratio (We found that decreasing the downsampling ratios of CNN layer may benefit the performance)
params.reg_window_max = 1e5;           % The maximum value of the regularization window
params.reg_window_min = 1e-3;           % The minimum value of the regularization window

% Detection parameters
params.refinement_iterations = 1;       % Number of iterations used to refine the resulting position in a frame
params.newton_iterations = 5;           % The number of Newton iterations used for optimizing the detection score
params.clamp_position = false;          % Clamp the target position to be inside the image

% Learning parameters
params.output_sigma_factor = 1/16;		% Label function sigma
params.temporal_regularization_factor = [15 15]; % The temporal regularization parameters

% ADMM parameters
params.max_iterations = [2 2];
params.init_penalty_factor = [1 1];
params.max_penalty_factor = [0.1, 0.1];
params.penalty_scale_step = [10, 10];

% Scale parameters for the translation model
params.number_of_scales = 5;            % Number of scales to run the detector
params.scale_step = 1.01;               % The scale factor

% Visualization
params.visualization = 1;               % Visualiza tracking and detection scores

% GPU
params.use_gpu = true;                 % Enable GPU or not
params.gpu_id = [];                     % Set the GPU id, or leave empty to use default

% Initialize
params.seq = seq;

% Run tracker
results = tracker(params);
