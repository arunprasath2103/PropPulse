import pytest
import torch
from backend.models.cnn_bilstm import CNNBiLSTM
from backend.models.cnn_lstm_inference import CNNLSTMInference
from backend.models.soft_sensor import SoftSensor
from backend.models.autoencoder import Autoencoder
from backend.training.train_offline import pinball_loss

def set_seed():
    torch.manual_seed(42)

# --- 1. Shape Verification & Forward Pass Stability ---

def test_offline_online_shapes():
    set_seed()
    offline_model = CNNBiLSTM(num_features=12).eval()
    online_model = CNNLSTMInference(num_features=12).eval()
    
    batch = torch.randn(32, 30, 12)
    
    with torch.no_grad():
        out10_off, out50_off, out90_off = offline_model(batch)
        out10_on, out50_on, out90_on = online_model(batch)
        
    assert out10_off.shape == (32, 1)
    assert out50_off.shape == (32, 1)
    assert out90_off.shape == (32, 1)
    
    assert out10_on.shape == (32, 1)
    assert out50_on.shape == (32, 1)
    assert out90_on.shape == (32, 1)

def test_batch_size_1():
    set_seed()
    offline_model = CNNBiLSTM(num_features=12).eval()
    online_model = CNNLSTMInference(num_features=12).eval()
    
    batch = torch.randn(1, 30, 12)
    
    with torch.no_grad():
        # Will pass successfully because models are in .eval() mode, skipping BatchNorm1d size-1 limitations
        out10, out50, out90 = offline_model(batch)
        assert out10.shape == (1, 1)
        
        out10_on, out50_on, out90_on = online_model(batch)
        assert out10_on.shape == (1, 1)

def test_soft_sensor_autoencoder_shapes():
    set_seed()
    soft_sensor = SoftSensor().eval()
    autoencoder = Autoencoder().eval()
    
    # Soft Sensor
    ss_batch = torch.randn(32, 14)
    with torch.no_grad():
        ss_out = soft_sensor(ss_batch)
    assert ss_out.shape == (32, 1)
    
    # Autoencoder
    ae_batch = torch.randn(32, 30, 12)
    with torch.no_grad():
        ae_out = autoencoder(ae_batch)
    assert ae_out.shape == (32, 360)

# --- 2. Pinball Loss Mathematical Verification ---

def test_pinball_loss_standard():
    y_true = torch.tensor([[100.0], [50.0], [20.0]])
    y_pred = torch.tensor([[90.0], [60.0], [20.0]])
    
    # 10th percentile weights: under=0.9, over=0.1
    # Err = [10, -10, 0]
    # Penalties: [10*0.9, 10*0.1, 0*0.9] -> [9.0, 1.0, 0.0]. Mean = 10.0 / 3
    loss_10 = pinball_loss(y_true, y_pred, under_weight=0.9, over_weight=0.1)
    expected_10 = torch.tensor(10.0 / 3.0)
    assert torch.isclose(loss_10, expected_10)

def test_pinball_loss_exact_predictions():
    y_true = torch.tensor([[100.0], [50.0]])
    y_pred = torch.tensor([[100.0], [50.0]])
    loss = pinball_loss(y_true, y_pred, under_weight=0.9, over_weight=0.1)
    assert loss.item() == 0.0

def test_pinball_loss_asymmetric_penalties():
    # 10th percentile head
    # Underestimation by 10 (err = +10)
    y_true_under = torch.tensor([[100.0]])
    y_pred_under = torch.tensor([[90.0]])
    loss_under_10 = pinball_loss(y_true_under, y_pred_under, under_weight=0.9, over_weight=0.1).item()
    
    # Overestimation by 10 (err = -10)
    y_true_over = torch.tensor([[100.0]])
    y_pred_over = torch.tensor([[110.0]])
    loss_over_10 = pinball_loss(y_true_over, y_pred_over, under_weight=0.9, over_weight=0.1).item()
    
    assert torch.isclose(torch.tensor(loss_under_10), torch.tensor(9 * loss_over_10))
    
    # 90th percentile head
    loss_under_90 = pinball_loss(y_true_under, y_pred_under, under_weight=0.1, over_weight=0.9).item()
    loss_over_90 = pinball_loss(y_true_over, y_pred_over, under_weight=0.1, over_weight=0.9).item()
    
    assert torch.isclose(torch.tensor(loss_over_90), torch.tensor(9 * loss_under_90))

# --- 3. Offline-to-Online Weight Transfer Compatibility ---

def test_weight_transfer_compatibility():
    set_seed()
    offline_model = CNNBiLSTM(num_features=12).eval()
    online_model = CNNLSTMInference(num_features=12).eval()
    
    # Verify they don't produce the same exact features randomly initially
    batch = torch.randn(8, 30, 12)
    batch_permuted = batch.permute(0, 2, 1)
    
    with torch.no_grad():
        x_on_initial = online_model.conv1(batch_permuted)
        x_off_initial = offline_model.conv1(batch_permuted)
    assert not torch.equal(x_on_initial, x_off_initial)
    
    # Extract and transfer weights
    online_model.conv1.load_state_dict(offline_model.conv1.state_dict())
    online_model.bn1.load_state_dict(offline_model.bn1.state_dict())
    online_model.conv2.load_state_dict(offline_model.conv2.state_dict())
    online_model.bn2.load_state_dict(offline_model.bn2.state_dict())
    
    with torch.no_grad():
        # Offline intermediate features
        x_off = offline_model.conv1(batch_permuted)
        x_off = offline_model.relu1(x_off)
        x_off = offline_model.bn1(x_off)
        x_off = offline_model.conv2(x_off)
        x_off = offline_model.relu2(x_off)
        x_off = offline_model.bn2(x_off)
        
        # Online intermediate features
        x_on = online_model.conv1(batch_permuted)
        x_on = online_model.relu1(x_on)
        x_on = online_model.bn1(x_on)
        x_on = online_model.conv2(x_on)
        x_on = online_model.relu2(x_on)
        x_on = online_model.bn2(x_on)
        
    assert torch.equal(x_off, x_on)

# --- 4. Autoencoder Bottleneck Constraint ---

def test_autoencoder_bottleneck():
    set_seed()
    autoencoder = Autoencoder().eval()
    batch = torch.randn(8, 30, 12)
    
    with torch.no_grad():
        latent_z = autoencoder.encoder(batch)
        
    assert latent_z.shape == (8, 32)
