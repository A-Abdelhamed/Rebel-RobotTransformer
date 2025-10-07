# ================================================================
# Convert Orbax checkpoint (Flax/JAX) -> TensorFlow SavedModel (.pb)
# ================================================================
import os
import jax
import tensorflow as tf
from jax.experimental import jax2tf
from orbax.checkpoint import PyTreeCheckpointer

# Suppress verbose TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ================================================================
# 1. PASTE YOUR MODEL DEFINITION HERE
# ================================================================
# You MUST copy the full class definitions for RT1, ImageTokenizer,
# Transformer, EfficientNetWithFilm, etc., from your training
# script into this section.
#
# EXAMPLE:
#
# import flax.linen as nn
#
# class ImageTokenizer(nn.Module):
#     # ... full class code ...
#
# class Transformer(nn.Module):
#     # ... full class code ...
#
# class RT1(nn.Module):
#     # ... full class code ...
#
# ================================================================

# --- Placeholder for demonstration (REPLACE WITH YOUR ACTUAL MODEL CODE) ---
import flax.linen as nn
class RT1(nn.Module):
    vocab_size: int = 512
    num_image_tokens: int = 81
    num_action_tokens: int = 11
    @nn.compact
    def __call__(self, obs, act, train):
        batch_size = obs['image'].shape[0]
        seq_len = obs['image'].shape[1]
        time_step_tokens = self.num_image_tokens + self.num_action_tokens
        x = obs['image'].reshape((batch_size * seq_len, -1))
        x = nn.Dense(features=time_step_tokens * self.vocab_size)(x)
        return x.reshape((batch_size, seq_len, time_step_tokens, self.vocab_size))
# --- End of placeholder ---


# ================================================================
# 2. CONFIGURE PATHS AND MODEL PARAMETERS
# ================================================================
# --- IMPORTANT: Set this to the path of your saved Orbax checkpoint ---
CHECKPOINT_PATH = "/app/rt1_checkpoints/step_1000"  # <-- CHANGE THIS

# --- Set the desired output path for the TensorFlow model ---
EXPORT_DIR = "/app/rt1_saved_model"

# --- Model Hyperparameters (must match your trained model) ---
MODEL_PARAMS = {
    'num_image_tokens': 81,
    'num_action_tokens': 11,
    'layer_size': 256,
    'vocab_size': 512,
    'use_token_learner': True,
    'world_vector_range': (-2.0, 2.0)
}
print("Configuration set.")

# ================================================================
# 3. LOAD THE JAX MODEL AND RESTORE WEIGHTS
# ================================================================
# Initialize the model architecture
model = RT1(**MODEL_PARAMS)

# Restore the parameters from the Orbax checkpoint
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
checkpointer = PyTreeCheckpointer()
# We only need the 'params' and 'batch_stats' for inference
restored_params = checkpointer.restore(CHECKPOINT_PATH, item='params')
restored_batch_stats = checkpointer.restore(CHECKPOINT_PATH, item='batch_stats')
variables = {'params': restored_params, 'batch_stats': restored_batch_stats}
print("✅ JAX checkpoint loaded successfully.")

# ================================================================
# 4. DEFINE THE FORWARD PASS FOR CONVERSION
# ================================================================
# This function defines how to run the model for inference.
def forward_pass(obs):
    """A wrapper function for the model's forward pass for inference."""
    # For inference, the action dictionary can be zeros
    dummy_act = {
        "world_vector": jax.numpy.zeros((1, 15, 3)),
        "rotation_delta": jax.numpy.zeros((1, 15, 3)),
        "gripper_closedness_action": jax.numpy.zeros((1, 15, 1)),
        "base_displacement_vertical_rotation": jax.numpy.zeros((1, 15, 1)),
        "base_displacement_vector": jax.numpy.zeros((1, 15, 2)),
        "terminate_episode": jax.numpy.zeros((1, 15, 3), dtype=jax.numpy.int32),
    }
    # Run the model with the restored variables in inference mode
    return model.apply(
        variables, obs, dummy_act, train=False, rngs={"random": jax.random.PRNGKey(0)}
    )

# ================================================================
# 5. CONVERT THE JAX FUNCTION TO TENSORFLOW
# ================================================================
# Define the exact shape and type of the inputs the final model should expect.
# This MUST match the observation dictionary.
input_signature = [{
    "image": tf.TensorSpec(shape=(1, 15, 300, 300, 3), dtype=tf.float32, name='image'),
    "natural_language_embedding": tf.TensorSpec(shape=(1, 15, 512), dtype=tf.float32, name='natural_language_embedding'),
}]

print("Starting JAX to TensorFlow conversion...")
# Convert the JAX function into a TensorFlow-compatible function
tf_predict_function = jax2tf.convert(forward_pass, with_gradient=False)
print("✅ Conversion complete.")

# ================================================================
# 6. SAVE THE TENSORFLOW SAVEDMODEL
# ================================================================
# Wrap the function in a tf.Module for saving
class TFModelWrapper(tf.Module):
    def __init__(self, tf_func):
        super().__init__()
        self.tf_func = tf_func

    @tf.function(input_signature=input_signature)
    def __call__(self, obs):
        return self.tf_func(obs)

# Create an instance of the wrapper
tf_model_wrapper = TFModelWrapper(tf_predict_function)

# Save the model
print(f"Saving TensorFlow SavedModel to: {EXPORT_DIR}")
os.makedirs(EXPORT_DIR, exist_ok=True)
tf.saved_model.save(tf_model_wrapper, EXPORT_DIR)
print(f"✅ Model saved. You can find it at: {EXPORT_DIR}")

# ================================================================
# 7. (OPTIONAL) TEST THE EXPORTED MODEL
# ================================================================
print("\n🔍 Testing the newly saved model...")
loaded_model = tf.saved_model.load(EXPORT_DIR)

# Create a dummy observation dictionary that matches the input signature
dummy_obs = {
    "image": tf.zeros((1, 15, 300, 300, 3), dtype=tf.float32),
    "natural_language_embedding": tf.zeros((1, 15, 512), dtype=tf.float32),
}

# Run inference
output = loaded_model(dummy_obs)
print("✅ Test successful. Model output shape:", output.shape)