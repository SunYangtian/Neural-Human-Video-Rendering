
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader

def CreateDataLoader_new(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader_new
    data_loader = CustomDatasetDataLoader_new()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
