# ================================================================
# Convert Orbax checkpoint (Flax/JAX) → TensorFlow SavedModel (.pb)
# ================================================================

import os
import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import jax2tf
from orbax.checkpoint import PyTreeCheckpointer

# ================================================================
# 1. Import your model definition
# ================================================================
# Adjust the import path to match your TrainingCode.py location
from TrainingCode import ImageTokenizer  # or your top-level model class

# ================================================================
# 2. Restore parameters from Orbax checkpoint
# ================================================================
checkpoint_dir = "/app/rt1_checkpoints/step_100000"   # <--- change if needed
checkpointer = PyTreeCheckpointer()
params = checkpointer.restore(checkpoint_dir)
print("✅ Restored parameters from:", checkpoint_dir)

# ================================================================
# 3. Recreate the Flax model
# ================================================================
# You must use the same model architecture and hyperparameters as in training
model = ImageTokenizer(num_tokens=8, num_features=512)
print("✅ Model loaded:", model.__class__.__name__)

# ================================================================
# 4. Define the forward function for inference
# ================================================================
def apply_fn(x, context):
    """Forward pass for exporting."""
    return model.apply({"params": params}, x, context_input=context, train=False)

# ================================================================
# 5. Convert the model to TensorFlow using jax2tf
# ================================================================
# ⚠️ Adjust input shapes and sizes to match your training setup
example_batch_size = 1
example_seq_len = 15     # e.g. trajectory length
image_height = 300
image_width = 300
num_channels = 3
context_dim = 512

input_signature = [
    tf.TensorSpec([example_batch_size, example_seq_len, image_height, image_width, num_channels], tf.float32),
    tf.TensorSpec([example_batch_size, context_dim], tf.float32)
]

tf_func = tf.function(jax2tf.convert(apply_fn, with_gradient=False), input_signature=input_signature)
print("✅ Converted JAX → TensorFlow")

# ================================================================
# 6. Export as TensorFlow SavedModel (.pb)
# ================================================================
export_dir = "/exported_model_pb/"
os.makedirs(export_dir, exist_ok=True)
tf.saved_model.save(tf_func, export_dir)
print(f"✅ Exported model saved to: {export_dir}/saved_model.pb")

# ================================================================
# 7. (Optional) Test the exported model
# ================================================================
print("\n🔍 Testing exported model...")
loaded = tf.saved_model.load(export_dir)

# Create dummy data for testing
example_input = tf.random.normal([example_batch_size, example_seq_len, image_height, image_width, num_channels])
example_context = tf.random.normal([example_batch_size, context_dim])

# Run inference
output = loaded(example_input, example_context)
print("✅ Exported model output shape:", [t.shape for t in output if hasattr(t, "shape")])
