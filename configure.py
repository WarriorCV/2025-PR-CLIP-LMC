def get_default_config(data_name):
    cfg = {}
    cfg['dataset'] = data_name

    name = cfg['dataset']
    if name == 'cifar10_4n_v':
        # Autoencoder 网络结构与激活
        cfg['Autoencoder'] = {

            'arch1': [324, 1024, 1024, 1024, 256], 
            'arch2': [10, 64, 128, 256, 256],  
            'arch3': [128, 512, 1024, 1024, 256],

            'activations1': 'relu',
            'activations2': 'relu',
            'activations3': 'relu',
            'batchnorm': True,
        }
        # 训练参数
        cfg['training'] = {
            'batch_size':      1000,
            'epoch':           100,
            'pretrain_epoch':  50,
            'lr':              1e-3,
            'seed':            2,
            'dim':             256,
            'num':             4,   
        }
        cfg['print_num'] = 50


    elif name == 'cifar10_8n_v':
        cfg['Autoencoder'] = {

            'arch1': [324, 512, 1024, 1024, 256],
            'arch2': [10, 64, 128, 256, 256],
            'arch3': [128, 256, 512, 512, 256],

            'activations1': 'relu',
            'activations2': 'relu',
            'activations3': 'relu',
            'batchnorm': True,
        }
        # 训练参数
        cfg['training'] = {
            'batch_size':      1000,
            'epoch':           100,
            'pretrain_epoch':  50,
            'lr':              1e-3,
            'seed':            2,
            'dim':             256,
            'num':             8,    #
        }

        cfg['print_num'] = 50


    elif name == 'imagenet4n_v':

        cfg['Autoencoder'] = {
            'arch1': [1764, 512, 1024, 1024, 256],         
            'arch2': [10, 64, 128, 256, 256],
            'arch3': [128, 512, 1024, 1024, 256],    

            'activations1': 'relu',
            'activations2': 'relu',
            'activations3': 'relu',
            'batchnorm': True,
        }

        cfg['training'] = {
            'batch_size':      1000,
            'epoch':           100,
            'pretrain_epoch':  50,
            'lr':              1e-3,
            'seed':            2,
            'dim':             256,
            'num':             4,    
        }

        cfg['print_num'] = 50

    elif name == 'oxford_4n_x':

        cfg['Autoencoder'] = {
            'arch1': [1764, 512, 1024, 1024, 256],
            'arch2': [10, 64, 128, 256, 256], 
            'arch3': [128, 64, 128, 256, 256], 
            'activations1': 'relu',
            'activations2': 'relu',
            'activations3': 'relu',
            'batchnorm': True,
        }

        cfg['training'] = {
            'batch_size':      200,
            'epoch':           100,
            'pretrain_epoch':  50,
            'lr':              1e-3,
            'seed':            2,
            'dim':             256,
            'num':             4,    
        }
        cfg['print_num'] = 50      


    return cfg