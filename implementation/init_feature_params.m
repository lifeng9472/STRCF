function feature_params = init_feature_params(features, feature_info)

% Initialize some feature parameters.
num_features = length(features);
feature_params.compressed_dim_block = cell(num_features, 1);

for k = 1:num_features
    if ~isfield(features{k}.fparams, 'compressed_dim')
        features{k}.fparams.compressed_dim = features{k}.fparams.nDim;
    end
    
    feature_params.compressed_dim_block{k} = features{k}.fparams.compressed_dim(:);
end

feature_params.compressed_dim = cell2mat(feature_params.compressed_dim_block);