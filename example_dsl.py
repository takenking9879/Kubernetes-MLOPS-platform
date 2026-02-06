from spark_feature_pipeline import Pipeline

# Load pipeline from YAML
pipeline = Pipeline.from_yaml("pipeline_config.yaml")

# Fit on training data
model = pipeline.fit(train_df)

# Transform new data
test_transformed = model.transform(test_df)

# Select final features
final_features = model.select_features(test_transformed)

# Save for later use
model.save("./pipeline_artifacts")