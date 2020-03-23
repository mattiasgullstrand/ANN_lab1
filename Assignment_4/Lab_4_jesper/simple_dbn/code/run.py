from util import *
from rbm import RestrictedBoltzmannMachine
from dbn import DeepBeliefNet

if __name__ == "__main__":
    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ref_test_indexes = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7] # contains indexes for all the digits in test dataset
    # print(np.argmax(test_lbls[ref_test_indexes, :], axis=1)) # check the labels of the ref indexes
    # exit()

    ''' restricted boltzmann machine '''
    '''
    print ("\nStarting a Restricted Boltzmann Machine..")
    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                      ndim_hidden=500,
                                      is_bottom=True,
                                      image_size=image_size,
                                      is_top=False,
                                      n_labels=10,
                                      batch_size=10
    )
    
    
    
    rbm.cd1(visible_trainset=train_imgs, n_iterations=3000)
    fig, axs = plt.subplots(2,5,figsize=(5,2))#,constrained_layout=True)
    plt.subplots_adjust(left=0,bottom=0,right=1,top=1,wspace=0,hspace=0)
    for x in range(2):
        for y in range(5):
            axs[x,y].set_xticks([])
            axs[x,y].set_yticks([])
            sample = test_imgs[ref_test_indexes[x*5+y],:]
            for _ in range(1000):
                sample = rbm.get_v_given_h(rbm.get_h_given_v(sample)[1])[1]
            sample = rbm.get_v_given_h(rbm.get_h_given_v(sample)[1])[0]
            axs[x,y].imshow(sample.reshape(image_size), cmap="bwr", vmin=-1, vmax=1, interpolation=None)
    plt.show()
    # exit()
    '''

    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=20
    )

    print('############ Greedy layer-wise training #############')

    dbn.train_greedylayerwise(vis_trainset=train_imgs,lbl_trainset=train_lbls, n_iterations=30000)

    #dbn.recognize(train_imgs, train_lbls)
    #
    print('############### DBN recognize ###############')
    print(test_imgs.shape)
    dbn.recognize(test_imgs, test_lbls)
    #exit()
    # exit()

    print('############# DBN generate ##################')
    #for digit in range(10):
    #     digit_1hot = np.zeros(shape=(1,10))
    #     digit_1hot[0,digit] = 1
    #     dbn.generate(digit_1hot, name="rbms")


    print('############### Fine-tune wake-sleep training ####################')

    dbn.train_wakesleep_finetune(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=30000)

    #dbn.recognize(train_imgs, train_lbls)

    print('################# DBN recognize #######################')
    dbn.recognize(test_imgs, test_lbls)
    #
    print('################## DBN generate #################')
    #for digit in range(10):
    #    digit_1hot = np.zeros(shape=(1,10))
    #    digit_1hot[0,digit] = 1
    #    dbn.generate(digit_1hot, name="dbn")
