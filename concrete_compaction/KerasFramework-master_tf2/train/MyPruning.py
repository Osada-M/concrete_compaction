# import tensorflow as tf
import numpy as np

from colorPrint import Cprint as cp


class MyPruning:
    """
    ( >o<) < Using SSIM !
    """
    
    is_print_details = False
    debug = False

    
    @staticmethod
    def tune_print(index:int, name:str, before:str="-", after:str=None):
        
        if not MyPruning.is_print_details + MyPruning.debug: return
        
        if after is not None:
            string = f"{index} : {name}{' '*(30 - len(name))} : {before} -> {after}"
        else:
            string = f"{index} : ( {name} ){' '*(26 - len(name))} : {before}"

        print(string)
        

    @staticmethod
    def calc_ssim(img_i, img_j):
        
        k_1 = 0.01
        k_2 = 0.03
        
        avg_i = np.mean(img_i)
        avg_j = np.mean(img_j)
        
        var_i = np.var(img_i)
        var_j = np.var(img_j)
        
        cov = np.mean((img_i - avg_i) * (img_j - avg_j))
        # cov = np.mean(np.cov(img_i, img_j))
        
        luminance = (2 * avg_i * avg_j + k_1) / ((avg_i ** 2) + (avg_j ** 2) + k_1)
        contrast = 1 / ((var_i ** 2) + (var_j ** 2) + k_2)
        structure = 2 * cov + k_2
        
        ssim = abs(luminance * contrast * structure)
        
        return ssim


    @staticmethod
    def reduce_near_zero(weights, next_shape):
        """
        Reduce the filter which has weight nearly zero.
        """
        
        sum_w = np.sum(abs(weights), axis=(0, 1))           
        filters = weights.shape[2:]
        
        next = np.argsort(np.sum(sum_w, axis=0)) >= (filters[1] - next_shape)
        next = [i for i in range(filters[1]) if next[i]]
        
        return next, sum_w, sum_w


    @staticmethod
    def reduce_ssim(weights, next_shape):
        """
        Reduce thr filter which is the most similar to many other filters by SSIM.
        """
        
        size = weights.shape[-1]
        ssim_matrix = np.zeros((size, size))
        vals = np.zeros((size, size))
        
        filters = np.sum(abs(weights), axis=(2))
        
        for i in range(size):
            vals[i] = np.zeros(size)
            for j in range(size):
                if (i == j):
                    vals[i][j] -= 1
                else:
                    vals[i][j] += MyPruning.calc_ssim(filters[:, :, i], filters[:, :, j])
            
            ssim_matrix[i] += np.identity(size)[np.argsort(vals[i])[-1]]
            ssim_matrix[i] += 0.1 * np.identity(size)[np.argsort(vals[i])[-2]]
            ssim_matrix[i] += 0.01 * np.identity(size)[np.argsort(vals[i])[-3]]
            ssim_matrix[i] += 0.001 * np.identity(size)[np.argsort(vals[i])[-4]]
        
        include_img = np.argsort(np.sum(ssim_matrix, axis=0))[:next_shape]
        next = [i for i in range(size) if i in include_img]
        
        return next, ssim_matrix, vals
    
    
    @staticmethod
    def reduce_error(weights, next_shape, dim=1):
        """
        Reduce thr filter which is the most similar to many other filters by MeanError.
        """
        
        size = weights.shape[-1]
        mse_matrix = np.zeros((size, size))
        vals = np.zeros((size, size))
        
        filters = np.sum(abs(weights), axis=(2))
        
        for i in range(size):
            vals[i] = np.zeros(size)
            for j in range(size):
                if (i == j):
                    vals[i][j] += 1
                else:
                    vals[i][j] -= np.sum(np.power(np.abs(filters[:, :, i] - filters[:, :, j]), dim))
            
            if np.min(vals[i]):
                vals[i][vals[i] == 1] += np.min(vals[i]) - 2
                vals[i] -= np.min(vals[i])
                # vals[i] /= np.min(vals[i])
                # vals[i] += 1
            else:
                vals[i][vals[i] == 1] -= 2
            
            mse_matrix[i] += np.identity(size)[np.argsort(vals[i])[-1]]
            mse_matrix[i] += 0.1 * np.identity(size)[np.argsort(vals[i])[-2]]
            mse_matrix[i] += 0.01 * np.identity(size)[np.argsort(vals[i])[-3]]
            mse_matrix[i] += 0.001 * np.identity(size)[np.argsort(vals[i])[-4]]
        
        include_img = np.argsort(np.sum(mse_matrix, axis=0))[:next_shape]
        next = [i for i in range(size) if i in include_img]
        
        return next, mse_matrix, vals


    @staticmethod
    def get_graph(model):
        """
        Get the graph of I/O relationship between each layer.
        """
        
        graph = dict()
        shape = dict()
        
        config = model.get_config()["layers"]
        for i, layer in enumerate(config):
            name = layer["config"]["name"]
            behind = layer["inbound_nodes"]

            if len(behind):
                key = behind[0][0][0]
                graph[name] = key
            
                if ("concat" in name):
                    shape[name] = shape[key] * 2
                elif ("add" in name):
                    shape[name] = shape[key]
                else:
                    shape[name] = model.layers[i].input.shape[-1]
            else:
                shape[name] = model.input.shape[-1]
            
        return graph, shape


    @staticmethod
    def prune_tuning(model_before, model_after, reduce="ssim"):
        """
        Excute filter-wise pruning.
        """
        
        cp.cprint(f"Reduce type : {reduce}", "pink")
        
        if MyPruning.debug:
            cp.cprint(f"Available debug mode", "pink")
            return MyPruning.prune_tuning_debug(model_before, model_after, reduce)
        
        # reduce_const = 1 - reduce_const
        graph, _ = MyPruning.get_graph(model_before)
        _, shape = MyPruning.get_graph(model_after)
        
        layer_length = len(graph)
        
        prune_indices = dict()
        next = list(range(model_before.input.shape[-1]))
        prune_indices[model_before.layers[0].name] = next.copy()
        
        print()
        for i, (l_be, l_af) in enumerate(zip(model_before.layers, model_after.layers)):
            
            ## For Conv2D.
            if ("conv" in l_be.name):

                ## Take model_before's weights.
                weights = np.array(l_be.weights[0])
                w_af = np.array(l_af.weights[0])
                
                if (weights.shape[-1] > 4):
                    next_shape = w_af.shape[-1]
                    ## Calculate indices of the layer has not unique weights.
                    if (reduce == "ssim"):
                        next, *_ = MyPruning.reduce_ssim(weights, next_shape)
                    elif(reduce == "error"):
                        next, *_ = MyPruning.reduce_error(weights, next_shape, dim=1)
                    elif(reduce == "square-error"):
                        next, *_ = MyPruning.reduce_error(weights, next_shape, dim=2)
                    elif(reduce == "zero"):
                        next, *_ = MyPruning.reduce_near_zero(weights, next_shape)
                    else:
                        return
                else:
                    next = list(range(w_af.shape[-1]))
                
                ## Record indices.
                prune_indices[l_be.name] = next.copy()
                behind = prune_indices[graph[l_be.name]].copy()
                
                ## Cut and set weights.
                cut_buf = weights[:, :, behind]
                cut_w = cut_buf[:, :, :, next]
                
                include_info = len(l_af.weights) > 1
                if include_info:
                    w_info = l_af.weights[1]
                    l_af.set_weights([cut_w, w_info])
                else:
                    l_af.set_weights([cut_w])
                
                MyPruning.tune_print(i, l_be.name, weights.shape, cut_w.shape)
                
            
            ## For Classifier Conv2D.
            elif ("classifier" in l_be.name):
                
                weights = np.array(l_be.weights[0])
                behind = prune_indices[graph[l_be.name]].copy()
                
                ## Cut and set weights.
                cut_w = weights[:, :, behind]
                w_info = l_af.weights[1]
                l_af.set_weights([cut_w, w_info])
                
                MyPruning.tune_print(i, l_be.name, weights.shape, cut_w.shape)


            ## For BatchNormalization.
            elif ("batch_normalization" in l_be.name):

                # length = int(np.ceil(shape[l_be.name] * reduce_const))
                length = shape[l_af.name]
                if (len(next) != length):
                    next = list(range(length))

                w = np.array(l_be.weights)
                cut_w = w[:, next]
                
                prune_indices[l_be.name] = next.copy()
                l_af.set_weights(cut_w)
                
                MyPruning.tune_print(i, l_be.name, w.shape, cut_w.shape)


            ## For Parametric ReLU.
            elif ("p_re" in l_be.name):

                length = shape[l_af.name]
                if (len(next) != length):
                    next = list(range(length))

                w = np.array(l_be.weights)
                cut_w = w[:, :, :, next]
                
                prune_indices[l_be.name] = next.copy()
                l_af.set_weights(cut_w)
                
                MyPruning.tune_print(i, l_be.name, w.shape, cut_w.shape)


            ## For Concatenate.
            elif ("concat" in l_be.name):
                
                # length = int(np.ceil(shape[l_be.name] * reduce_const / 2) * 2)
                length = shape[l_af.name]
                prune_indices[l_be.name] = list(range(length))

                MyPruning.tune_print(i, l_be.name, length)
            
            
            else:

                l_af.set_weights(l_be.weights)
                prune_indices[l_be.name] = next.copy()
                
                MyPruning.tune_print(i, l_be.name)
                
            if MyPruning.is_print_details + MyPruning.debug: continue
            
            cp.cprint(f"\033[1Apruning... ( {i+1} / {layer_length} layers )", "cyan")
            

        return model_after
    
    
    @staticmethod
    def prune_tuning_debug(model_before, model_after, reduce="ssim"):
        """
        For debuging.
        """
        
        # reduce_const = 1 - reduce_const
        graph, _ = MyPruning.get_graph(model_before)
        _, shape = MyPruning.get_graph(model_after)
        
        layer_length = len(graph)
        
        prune_indices = dict()
        next = list(range(model_before.input.shape[-1]))
        prune_indices[model_before.layers[0].name] = next.copy()
        
        print()
        for i, (l_be, l_af) in enumerate(zip(model_before.layers, model_after.layers)):
            
            ## For Conv2D.
            if ("conv" in l_be.name):

                ## Take model_before's weights.
                weights = np.array(l_be.weights[0])
                w_af = np.array(l_af.weights[0])
                
                if (weights.shape[-1] > 4):
                    next_shape = w_af.shape[-1]
                    ## Calculate indices of the layer has not unique weights.
                    if (reduce == "ssim"):
                        next, matrix, vals = MyPruning.reduce_ssim(weights, next_shape)
                    elif(reduce == "error"):
                        next, matrix, vals = MyPruning.reduce_error(weights, next_shape, dim=1)
                    elif(reduce == "square-error"):
                        next, matrix, vals = MyPruning.reduce_error(weights, next_shape, dim=2)
                    elif(reduce == "zero"):
                        next, matrix, vals = MyPruning.reduce_near_zero(weights, next_shape)
                    else:
                        return
                else:
                    next = list(range(w_af.shape[-1]))
                
                ## Record indices.
                prune_indices[l_be.name] = next.copy()
                behind = prune_indices[graph[l_be.name]].copy()
                
                ## Cut and set weights.
                cut_buf = weights[:, :, behind]
                cut_w = cut_buf[:, :, :, next]
                
                include_info = len(l_af.weights) > 1
                if include_info:
                    w_info = l_af.weights[1]
                    l_af.set_weights([cut_w, w_info])
                else:
                    l_af.set_weights([cut_w])
                
                MyPruning.tune_print(i, l_be.name, weights.shape, cut_w.shape)
                
                yield i, l_be.name, weights, next, matrix, vals
            
            
            ## For Classifier Conv2D.
            elif ("classifier" in l_be.name):
                
                weights = np.array(l_be.weights[0])
                behind = prune_indices[graph[l_be.name]].copy()
                
                ## Cut and set weights.
                cut_w = weights[:, :, behind]
                w_info = l_af.weights[1]
                l_af.set_weights([cut_w, w_info])
                
                MyPruning.tune_print(i, l_be.name, weights.shape, cut_w.shape)


            ## For BatchNormalization.
            elif ("batch_normalization" in l_be.name):

                # length = int(np.ceil(shape[l_be.name] * reduce_const))
                length = shape[l_af.name]
                if (len(next) != length):
                    next = list(range(length))

                w = np.array(l_be.weights)
                cut_w = w[:, next]
                
                prune_indices[l_be.name] = next.copy()
                l_af.set_weights(cut_w)
                
                MyPruning.tune_print(i, l_be.name, w.shape, cut_w.shape)


            ## For Parametric ReLU.
            elif ("p_re" in l_be.name):

                length = shape[l_af.name]
                if (len(next) != length):
                    next = list(range(length))

                w = np.array(l_be.weights)
                cut_w = w[:, :, :, next]
                
                prune_indices[l_be.name] = next.copy()
                l_af.set_weights(cut_w)
                
                MyPruning.tune_print(i, l_be.name, w.shape, cut_w.shape)


            ## For Concatenate.
            elif ("concat" in l_be.name):
                
                # length = int(np.ceil(shape[l_be.name] * reduce_const / 2) * 2)
                length = shape[l_af.name]
                prune_indices[l_be.name] = list(range(length))

                MyPruning.tune_print(i, l_be.name, length)
            
            
            else:

                l_af.set_weights(l_be.weights)
                prune_indices[l_be.name] = next.copy()
                
                MyPruning.tune_print(i, l_be.name)
                
            if MyPruning.is_print_details + MyPruning.debug: continue
            
            cp.cprint(f"\033[1Apruning... ( {i+1} / {layer_length} layers )", "cyan")
            
        
        yield -1, None, None, None, None, None

