import torch
import torch.nn as nn
import copy
import gc
import argparse
import cv2

class Evaluator():
    
    
    def gram_matrix(self, input):
        a, b, c, d = input.size()
        input = input.view(b, c * d)
        G = torch.mm(input, input.t())
        return G.div(b * c * d)
    
    def calculate_content_fidelity(self, c, x, model):
        
        total_fidelity = 0
        num_layers = 0
        cos = nn.CosineSimilarity()
        with torch.no_grad():
            for layer in model:
                c_features = layer(c)
                x_features = layer(x)
                output = cos(c_features, x_features).mean()

                total_fidelity += output
                num_layers += 1
                c = layer(c)
                x = layer(x)
        return total_fidelity / num_layers

    
    def calculate_global_effects(self, x, s, model, output_img_filepath, style_img_filepath):
        print(output_img_filepath)
        print(style_img_filepath)
        
        # Trying Color Histogram Correlation for Global Colors
        x_image = cv2.imread(output_img_filepath)
        s_image = cv2.imread(style_img_filepath)
        
        x_hist = cv2.calcHist([x_image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        x_hist = cv2.normalize(x_hist, x_hist).flatten()
        
        s_hist = cv2.calcHist([s_image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        
        correlation = cv2.compareHist(x_hist, s_hist, cv2.HISTCMP_CORREL)
        
        cos = nn.CosineSimilarity()
        with torch.no_grad():
            # Holistic Texture (HT)
            holistic_texture = 0
            num_layers = 0
            for layer in model:
                s_gram = torch.unsqueeze(self.gram_matrix(layer(s)), 1)
                x_gram = torch.unsqueeze(self.gram_matrix(layer(x)), 1)
                holistic_texture += cos(s_gram, x_gram).mean()
                num_layers += 1
                s = layer(s)
                x = layer(x)
        holistic_texture = holistic_texture / num_layers
        
        return (correlation + holistic_texture) / 2
 
    def extract_patches(self, tensor):
        unfold_func = nn.Unfold(kernel_size=(32,32), stride=16)      
        return unfold_func(tensor)
        
    def calculate_local_patterns(self, x, s, model):

        ## Using batches for memory
        num_layers = 0
        cross_correlation_total = 0
        category_comp_total = 0
        cos = nn.CosineSimilarity()
        
        with torch.no_grad():
            for layer in model:
                        
                gc.collect()
                torch.cuda.empty_cache()

                x_patches = self.extract_patches(layer(x))
                s_patches = self.extract_patches(layer(s))

                best_patch_matches = []
                best_value = 0

                for patch in x_patches:
                    patch = torch.unsqueeze(patch, 1)
                    best_patch = torch.zeros_like(patch)
                    best_value = 0

                    for ref_patch in s_patches:
                        gc.collect()
                        torch.cuda.empty_cache()

                        ref_patch =  torch.unsqueeze(ref_patch, 1)
                        temp_val = cos(patch, ref_patch).mean()
                        if (best_value < temp_val):
                            best_patch = ref_patch
                            best_value = temp_val
                            if (best_patch_matches.count(best_patch) == 0):
                                best_patch_matches.append(best_patch)

                    cross_correlation_total += best_value

                    gc.collect()
                    torch.cuda.empty_cache()

                #print(len(best_patch_matches))
                #print(s_patches.shape)
                a, b, c = s_patches.shape
                category_comp_total += (len(best_patch_matches) / a)
                num_layers += 1
                x = layer(x)
                s = layer(s)
        
        q, w, e = x_patches.shape
        LP_1 = cross_correlation_total.item() / (q * num_layers)
        LP_2 = category_comp_total / num_layers
        return (LP_1 + LP_2) / 2