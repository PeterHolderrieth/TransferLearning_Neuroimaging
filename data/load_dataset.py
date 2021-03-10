import sys 
from data.oasis.load_oasis3 import give_oasis_data
from data.abide.load_abide import give_abide_data
from data.ixi.load_ixi import give_ixi_data

def give_dataset(data_set,data_type,batch_size,num_workers,shuffle,debug,preprocessing,task,share):      
        if data_set=='oasis':
            return give_oasis_data(data_type, batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    shuffle=shuffle,
                                                    debug=debug,
                                                    preprocessing=preprocessing,
                                                    task=task,
                                                    share=share)

        elif data_set=='abide':
            return give_abide_data(data_type, batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    shuffle=shuffle,
                                                    debug=debug,
                                                    preprocessing=preprocessing,
                                                    task=task,
                                                    share=share)

        elif data_set=='ixi':
            return give_ixi_data(data_type, batch_size=batch_size,
                                                    num_workers=num_workers,
                                                    shuffle=shuffle,
                                                    debug=debug,
                                                    preprocessing=preprocessing,
                                                    task=task,
                                                    share=share)
        else: 
            sys.exit("Unknown data set.")