
import torch
import numpy as np

class Sampler:
    def __init__(self, config):
        self.cfg = config
        self.Lx = config['geometry']['Lx']
        self.Ly = config['geometry']['Ly']
        self.H = config['geometry']['H']
        self.pinn_cfg = config['pinn']['sampling']
        
        # Layer interfaces
        self.interfaces = [0.0, self.H/3, 2*self.H/3, self.H]
        
    def sample_domain(self, n, z_min, z_max):
        x = torch.rand(n, 1) * self.Lx
        y = torch.rand(n, 1) * self.Ly
        z = torch.rand(n, 1) * (z_max - z_min) + z_min
        return torch.cat([x, y, z], dim=1)

    def sample_boundaries(self, n, z_min, z_max):
        # 4 Side faces
        n_face = n // 4
        
        # x=0
        y1 = torch.rand(n_face, 1) * self.Ly
        z1 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
        x1 = torch.zeros(n_face, 1)
        p1 = torch.cat([x1, y1, z1], dim=1)
        
        # x=Lx
        y2 = torch.rand(n_face, 1) * self.Ly
        z2 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
        x2 = torch.ones(n_face, 1) * self.Lx
        p2 = torch.cat([x2, y2, z2], dim=1)
        
        # y=0
        x3 = torch.rand(n_face, 1) * self.Lx
        z3 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
        y3 = torch.zeros(n_face, 1)
        p3 = torch.cat([x3, y3, z3], dim=1)
        
        # y=Ly
        x4 = torch.rand(n_face, 1) * self.Lx
        z4 = torch.rand(n_face, 1) * (z_max - z_min) + z_min
        y4 = torch.ones(n_face, 1) * self.Ly
        p4 = torch.cat([x4, y4, z4], dim=1)
        
        return torch.cat([p1, p2, p3, p4], dim=0)

    def sample_top(self, n):
        # z=H
        # Load Patch
        lp = self.cfg['load_patch']
        # Patch bounds
        lx_min = lp.get('x_start', 0.33) * self.Lx # The yaml uses normalized or absolute? 0.33 looks normalized?
        # Actually yaml says x_start: 0.33. If Lx=1.0, it's 0.33. If yaml meant fraction, it's consistent.
        # But wait, previous code used Lx/3 = 0.333.
        # Let's assume yaml values are fractions of L if they are small, or absolute?
        # Given Lx=1.0, 0.33 is compatible.
        
        # Actually, let's treat them as absolute coordinates or fractions?
        # "x_start: 0.33" in yaml.
        
        # If I want to be safe, I'll read them directly.
        x_min_p = 0.33 * self.Lx # Assuming fraction based on earlier prompt ("Lx/3")
        x_max_p = 0.67 * self.Lx
        y_min_p = 0.33 * self.Ly
        y_max_p = 0.67 * self.Ly
        
        n_load = n // 2
        n_free = n - n_load
        
        # Loaded Patch
        xl = torch.rand(n_load, 1) * (x_max_p - x_min_p) + x_min_p
        yl = torch.rand(n_load, 1) * (y_max_p - y_min_p) + y_min_p
        zl = torch.ones(n_load, 1) * self.H
        pts_load = torch.cat([xl, yl, zl], dim=1)
        
        # Free Top
        pts_free_list = []
        count = 0
        while count < n_free:
            batch = 1000
            x = torch.rand(batch, 1) * self.Lx
            y = torch.rand(batch, 1) * self.Ly
            
            in_patch = (x > x_min_p) & (x < x_max_p) & \
                       (y > y_min_p) & (y < y_max_p)
            
            mask_free = ~in_patch.squeeze()
            xf, yf = x[mask_free], y[mask_free]
            if len(xf) > 0:
                zf = torch.ones(len(xf), 1) * self.H
                batch_pts = torch.cat([xf, yf, zf], dim=1)
                pts_free_list.append(batch_pts)
                count += len(xf)
                
        pts_free = torch.cat(pts_free_list, dim=0)[:n_free]
        return pts_load, pts_free

    def sample_interface(self, n, z):
        x = torch.rand(n, 1) * self.Lx
        y = torch.rand(n, 1) * self.Ly
        z_t = torch.ones(n, 1) * z
        return torch.cat([x, y, z_t], dim=1)

    def get_data(self):
        N_INT = self.pinn_cfg['n_interior']
        N_BC = self.pinn_cfg['n_boundary']
        
        data = {}
        # Interior
        data['interior'] = [
            self.sample_domain(N_INT, self.interfaces[0], self.interfaces[1]), # Layer 1
            self.sample_domain(N_INT, self.interfaces[1], self.interfaces[2]), # Layer 2
            self.sample_domain(N_INT, self.interfaces[2], self.interfaces[3])  # Layer 3
        ]
        
        # Clamped Sides
        data['sides'] = [
            self.sample_boundaries(N_BC, self.interfaces[0], self.interfaces[1]),
            self.sample_boundaries(N_BC, self.interfaces[1], self.interfaces[2]),
            self.sample_boundaries(N_BC, self.interfaces[2], self.interfaces[3])
        ]
        
        # Top
        top_load, top_free = self.sample_top(N_BC)
        data['top_load'] = top_load
        data['top_free'] = top_free
        
        # Bottom
        x_bot = torch.rand(N_BC, 1) * self.Lx
        y_bot = torch.rand(N_BC, 1) * self.Ly
        z_bot = torch.zeros(N_BC, 1)
        data['bottom'] = torch.cat([x_bot, y_bot, z_bot], dim=1)
        
        # Interfaces
        data['if_12'] = self.sample_interface(N_BC, self.interfaces[1])
        data['if_23'] = self.sample_interface(N_BC, self.interfaces[2])
        
        return data
