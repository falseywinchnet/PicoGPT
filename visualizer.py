
import numpy as np
import torch
import torch.nn.functional as F
import ipywidgets as widgets
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
import io
import math
import time

class FastMatrixDashboard:
    def __init__(self, batch_size, seq_len, itos=None, cell_w=10, cell_h=16, target_fps=20):
        """
        High-Performance Vectorized Dashboard by joshuah rainstar, {your name here if you make edits}
        joshuah.rainstar@gmail.com
        --your email here--
        """
        self.target_cells = batch_size * seq_len
        self.itos = itos
        
        # --- 1. Geometry ---
        self.rows = int(math.sqrt(self.target_cells / 5))
        self.cols = int(np.ceil(self.target_cells / self.rows))
        self.n_cells = self.rows * self.cols
        
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.width = self.cols * self.cell_w
        self.height = self.rows * self.cell_h
        self.stats_height = 40
        self.total_height = self.height + self.stats_height

        # --- 2. Font & Atlas Setup (The Speedup) ---
        # We render all chars to a numpy bank once.
        try:
            self.font = ImageFont.truetype("DejaVuSansMono.ttf", 11)
        except:
            self.font = ImageFont.load_default()

        # Create Atlas: Shape (256, H, W) - Pre-render ASCII 0-255
        # We use a mask: 1.0 where text is, 0.0 where background is.
        self.atlas = np.zeros((256, self.cell_h, self.cell_w), dtype=np.float32)
        
        temp_img = Image.new("L", (self.cell_w, self.cell_h))
        temp_draw = ImageDraw.Draw(temp_img)
        
        for i in range(256):
            char = chr(i) if 32 <= i <= 126 else "?"
            # Custom replacements for visibility
            if i == 10: char = "¶"  # Newline
            if i == 9:  char = "→"  # Tab
            if i == 32: char = "·"  # Space
            
            temp_draw.rectangle((0, 0, self.cell_w, self.cell_h), fill=0)
            temp_draw.text((0, 0), char, font=self.font, fill=255)
            # Normalize to 0..1
            self.atlas[i] = np.array(temp_img, dtype=np.float32) / 255.0

        # --- 3. Token Lookup Optimization ---
        # Map vocab IDs -> Atlas IDs (0-255)
        if self.itos:
            vocab_size = max(self.itos.keys()) + 1
            self.vocab_map = np.zeros(vocab_size, dtype=int)
            for k, v in self.itos.items():
                # Take first char ord or ?
                char_code = ord(v[0]) if len(v) > 0 else 63
                # Ensure range
                if not (0 <= char_code <= 255): char_code = 63 
                self.vocab_map[k] = char_code
        else:
            self.vocab_map = None # Direct mapping

        # --- 4. Simulation State (Numpy Arrays) ---
        # Instead of a list of chars, we store indices and colors
        self.state_indices = np.full(self.n_cells, 32, dtype=int) # 32 = Space
        self.state_colors = np.zeros((self.n_cells, 3), dtype=np.float32) + 40.0
        self.freshness = np.zeros(self.n_cells, dtype=np.float32)
        
        self.ewma_loss = None
        self.step = 0
        self.last_render_time = 0
        self.min_render_interval = 1.0 / target_fps

        # --- 5. Widget ---
        self.out_widget = widgets.Image(format='png', width=self.width, height=self.total_height)
        self.layout = widgets.VBox([self.out_widget])

    def render(self):
        display(self.layout)

    def update(self, yb, logits, loss_val):
        self.step += 1
        
        # Throttling: Don't render if we just rendered (keeps training loop fast)
        now = time.time()
        if now - self.last_render_time < self.min_render_interval:
            return
        self.last_render_time = now

        # --- 1. Tensor Ops (Fast) ---
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            p_max, preds = torch.max(probs, dim=-1)
            
            p_max = p_max.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()
            targets = yb.cpu().numpy().flatten()

        limit = min(len(p_max), self.n_cells)
        
        # --- 2. Vectorized Freshness Logic ---
        # Note: Operations are done on arrays, not loops
        is_correct = (preds[:limit] == targets[:limit]).astype(np.float32)
        self.freshness *= 0.92
        
        current_fresh = self.freshness[:limit]
        new_conf = p_max[:limit]
        
        # Mask: where to update
        update_mask = (new_conf > current_fresh) | (current_fresh < 0.10)
        
        # Update freshness
        self.freshness[:limit] = np.where(update_mask, new_conf, current_fresh)
        
        if not np.any(update_mask):
            return # Nothing visual changed significantly

        # --- 3. Vectorized Color & Char Mapping ---
        # Create target colors
        vals = new_conf * 255.0
        vals = np.maximum(50.0, vals)
        
        # R, G, B vectors
        r = (is_correct[:limit] * (vals * 0.5) + (1 - is_correct[:limit]) * vals)
        g = (is_correct[:limit] * vals + (1 - is_correct[:limit]) * (vals * 0.5))
        b = (is_correct[:limit] * (vals * 0.25))
        
        new_colors = np.stack([r, g, b], axis=1) # (limit, 3)
        
        # Map tokens to atlas indices
        if self.vocab_map is not None:
            # Safe lookup handling bounds
            safe_preds = np.clip(preds[:limit], 0, len(self.vocab_map)-1)
            safe_targets = np.clip(targets[:limit], 0, len(self.vocab_map)-1)
            
            token_indices = self.vocab_map[safe_preds]
            target_indices = self.vocab_map[safe_targets]
            
            # Fallback logic (vectorized): if OOV (mapped to '?'), use target
            # Assuming '?' is index 63.
            # A better heuristic for "OOV" in this optimized version might just be 
            # relying on the vocab_map. 
            # If strict OOV check is needed, we check if preds not in self.itos.
            # For speed, we trust the vocab_map handles the fallback.
        else:
            # If no itos, use raw ASCII
            token_indices = np.clip(preds[:limit], 32, 126)

        # Update state buffers
        self.state_indices[:limit] = np.where(update_mask, token_indices, self.state_indices[:limit])
        self.state_colors[:limit] = np.where(update_mask[:, None], new_colors, self.state_colors[:limit])

        # --- 4. Image Composition (The Heavy Lifting moved to Numpy) ---
        # 1. Retrieve Masks: (N, H, W)
        masks = self.atlas[self.state_indices] 
        
        # 2. Apply Colors: (N, H, W, 3)
        # Broadcast: (N, H, W) -> (N, H, W, 1) * (N, 1, 1, 3)
        grid_pixels = masks[..., None] * self.state_colors[:, None, None, :]
        
        # 3. Reshape to Grid Image (Rows, Cols, H, W, 3)
        # Pad if necessary to fill grid
        if grid_pixels.shape[0] < self.n_cells:
            padding = np.zeros((self.n_cells - grid_pixels.shape[0], self.cell_h, self.cell_w, 3))
            grid_pixels = np.concatenate([grid_pixels, padding], axis=0)

        grid_reshaped = grid_pixels.reshape(self.rows, self.cols, self.cell_h, self.cell_w, 3)
        
        # 4. Transpose to (Rows, H, Cols, W, 3) -> (Height, Width, 3)
        final_grid = grid_reshaped.transpose(0, 2, 1, 3, 4).reshape(self.height, self.width, 3)
        
        # Cast to uint8
        final_img_arr = np.clip(final_grid, 0, 255).astype(np.uint8)

        # --- 5. Stats Bar (PIL is fine here, it's small) ---
        # We create the stats bar separately and stack it
        if self.ewma_loss is None: self.ewma_loss = loss_val
        else: self.ewma_loss = 0.95 * self.ewma_loss + 0.05 * loss_val
        acc = np.mean(is_correct)

        stats_img = Image.new("RGB", (self.width, self.stats_height), (20, 20, 20))
        draw = ImageDraw.Draw(stats_img)
        # Stats Text
        draw.text((10, 10), f"STEP: {self.step}", font=self.font, fill=(200, 200, 200))
        draw.text((100, 10), f"LOSS: {loss_val:.4f}", font=self.font, fill=(255, 100, 100))
        draw.text((220, 10), f"EWMA: {self.ewma_loss:.4f}", font=self.font, fill=(255, 255, 0))
        draw.text((340, 10), f"ACC: {acc:.1%}", font=self.font, fill=(0, 255, 0))

        # --- 6. Final Combine & Compress ---
        # Convert Stats to numpy
        stats_arr = np.array(stats_img)
        
        # Vertical Stack
        full_frame = np.vstack((stats_arr, final_img_arr))
        
        # Convert to PNG using low compression for speed
        # PIL.Image.fromarray is zero-copy for uint8 usually
        img_obj = Image.fromarray(full_frame)
        
        with io.BytesIO() as output:
            # compress_level=1 is much faster than default (6)
            img_obj.save(output, format="PNG", compress_level=1) 
            self.out_widget.value = output.getvalue()

# Example Usage:
dashboard = FastMatrixDashboard(batch_size, block_size, itos=itos)
dashboard.render()
# In loop: dashboard.update(yb, logits, loss.item())
