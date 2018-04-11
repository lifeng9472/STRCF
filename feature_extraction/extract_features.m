function res_feat = extract_features(image, pos, scales, features, gparams, extract_info)

% Sample image patches at given position and scales. Then extract features
% from these patches.
% Requires that cell size and image sample size is set for each feature.

if ~iscell(features)
    error('Wrong input');
end

if ~isfield(gparams, 'use_gpu')
    gparams.use_gpu = false;
end
if ~isfield(gparams, 'data_type')
    gparams.data_type = zeros(1, 'single');
end
if nargin < 6
    % Find used image sample size
    extract_info = get_feature_extract_info(features);
end

num_features = length(features);
num_scales = length(scales);
num_sizes = length(extract_info.img_sample_sizes);

% Extract image patches
img_samples = cell(num_sizes,1);
for sz_ind = 1:num_sizes
    img_sample_sz = extract_info.img_sample_sizes{sz_ind};
    img_input_sz = extract_info.img_input_sizes{sz_ind};
    img_samples{sz_ind} = zeros(img_input_sz(1), img_input_sz(2), size(image,3), num_scales, 'uint8');
    for scale_ind = 1:num_scales
        img_samples{sz_ind}(:,:,:,scale_ind) = get_pixels(image, pos, round(img_sample_sz*scales(scale_ind)),img_input_sz);
    end
end

% Find the number of feature blocks and total dimensionality
num_feature_blocks = 0;
total_dim = 0;
for feat_ind = 1:num_features
    num_feature_blocks = num_feature_blocks + length(features{feat_ind}.fparams.nDim);
    total_dim = total_dim + sum(features{feat_ind}.fparams.nDim);
end

feature_map = cell(1, 1, num_feature_blocks);

% Extract feature maps for each feature in the list
ind = 1;
CNN_ind = -1;
for feat_ind = 1:num_features
    feat = features{feat_ind};
    
    % get the image patch index
    img_sample_ind = cellfun(@(sz) isequal(feat.img_sample_sz, sz), extract_info.img_sample_sizes);
    
    % do feature computation
    if feat.is_cell
        num_blocks = length(feat.fparams.nDim);
        feature_map(ind:ind+num_blocks-1) = feat.getFeature(img_samples{img_sample_ind}, feat.fparams, gparams);
        CNN_ind = ind;
    else
        num_blocks = 1;
        feature_map{ind} = feat.getFeature(img_samples{img_sample_ind}, feat.fparams, gparams);
    end
    
    ind = ind + num_blocks;
end

% Do CNN feature normalization per channel
if(CNN_ind ~= -1)
    CNN_feat = {feature_map{CNN_ind}};
    delta = 1e-7;
    
    for i = 1:numel(CNN_feat)
        min_ele = repmat(min(min(CNN_feat{i},[],2),[],1), [size(CNN_feat{i},1),size(CNN_feat{i},2),1,1]);
        max_ele = repmat(max(max(CNN_feat{i},[],2),[],1), [size(CNN_feat{i},1),size(CNN_feat{i},2),1,1]) + delta;
        CNN_feat{i} = ( CNN_feat{i} - min_ele ) ./ (max_ele - min_ele);
    end
end

% Combine the features. In particular, the hand-crafted features are concatnated together
% along the channels.
if(CNN_ind ~= -1)
    res_feat = cell(2, 1);
    res_feat{2} = CNN_feat{1};
end

% concatnate the grayscale and HOG features only when the input is the grayscale
% image.
if(CNN_ind ~=-1)
    if(num_features == 3)
        res_feat{1} = cat(3,feature_map{1}, feature_map{2});    
    else
        res_feat{1} = cat(3,feature_map{1}, feature_map{2}, feature_map{3});  
    end
else
    if(num_features == 3)
        res_feat{1} = cat(3,feature_map{1}, feature_map{2}, feature_map{3});  
    else
        res_feat{1} = cat(3,feature_map{1}, feature_map{2});    
    end
end

end