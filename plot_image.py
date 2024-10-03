import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np
from scipy.interpolate import griddata

def itep2D(data):
    # 准备插值的网格
    x = np.arange(64)
    y = np.arange(64)
    X, Y = np.meshgrid(x, y)
    limt = 0
    for imag in data:
        for chanel in imag: 
            # 识别值不为0的位置
            org_point_pos = (chanel > limt)
            int_point_pos = (chanel < limt)
            values = chanel[org_point_pos]
            points = np.array([X[org_point_pos], Y[org_point_pos]]).T
            # 使用 griddata 进行插值
            interpolated = griddata(points, values, (X, Y), method='nearest')
            # 将插值后的数据填回原数据，只替换值为0的点
            chanel[int_point_pos] = interpolated[int_point_pos]
    return data


# calculate the chi2 of mock
def calculate_mock_chi(data, epsilon=1e-3):
    input = data['input']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    noise = data['noise']
    noise = np.abs(noise) + epsilon
    chi1 = (input - pred_fore)**2/noise**2
    chi1 = np.mean(chi1, axis=(-3,-2,-1))
    chi2 = (input - pred_fore - pred_back)**2/noise**2
    chi2 = np.mean(chi2, axis=(-3,-2,-1))
    return chi1, chi2

def calculate_real_chi(data):
    input = data['input']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    wei = data['weight']
    norm = data['norm']
    residual = input - pred_fore
    chi1 = (residual/norm)**2*wei
    chi1 = np.mean(chi1, axis=(-3,-2,-1))
    residual = input - pred_fore - pred_back
    chi2 = (residual/norm)**2*wei
    chi2 = np.mean(chi2, axis=(-3,-2,-1))
    return chi1, chi2

def calculate_HST_chi(data):
    input = data['input']
    fore_image = data['fore_image']
    back_image = data['back_image']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    wei = data['weight']

    residual = input - pred_fore
    chi1 = (residual)**2*wei
    chi1 = np.mean(chi1, axis=(-3,-2,-1))
    residual = input - pred_fore - pred_back
    chi2 = (residual)**2*wei
    chi2 = np.mean(chi2, axis=(-3,-2,-1))

    residual = input - fore_image
    model_chi1 = (residual)**2*wei
    model_chi1 = np.mean(model_chi1, axis=(-3,-2,-1))
    residual = input - fore_image - back_image
    model_chi2 = (residual)**2*wei
    model_chi2 = np.mean(model_chi2, axis=(-3,-2,-1))

    return model_chi1, model_chi2, chi1, chi2


#================================================== ploting
def display_mock_image(data, size=2, save_name='pred_mock', one_band=False, epsilon=1e-3, intep=False):

    input = data['input']
    fore_image = data['fore_image']
    back_image = data['back_image']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    if intep:
        pred_fore = itep2D(pred_fore)
        pred_back = itep2D(pred_back)
    noise = data['noise']
    #RA, DEC = data['RA'], data['DEC']
    chi1, chi2 = calculate_mock_chi(data, epsilon)
    real_residual = input-fore_image-back_image
    pred_residual = input-pred_fore-pred_back
    image_list = [input, fore_image, pred_fore, back_image, pred_back, real_residual, pred_residual]
    fig, axes = plt.subplots(input.shape[0], len(image_list), figsize=(size*len(image_list), size*input.shape[0]), sharex=False, sharey=False)
    
    rad = 5
    for i in range(input.shape[0]):
        for j in range(len(image_list)):
            image = image_list[j][i]
            size = image.shape[-1]
            if one_band:
                RGB_image = image[0]
                norm = np.max(RGB_image[size//2-rad:size//2+rad,size//2-rad:size//2+rad])
                axes[i][j].imshow(RGB_image/norm)
            else:
                RGB_image = image[[3,2,1]]  # Conventionally R = i', G = r', and B = g'.
                norm = np.max(RGB_image[ : ,size//2-rad:size//2+rad,size//2-rad:size//2+rad]) 
                axes[i][j].imshow(np.transpose(RGB_image, (1, 2, 0))/norm)
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)

        #text1 = 'ra=' + str(round(RA[i],3)) + '\ndec=' + str(round(DEC[i],3))
        text1 = 'id=' + str(i+1)
        text2 = '$\chi_1^2=$'+str(round(chi1[i],2))
        text3 = '$\chi_2^2=$'+str(round(chi2[i],2))
        axes[i][0].text(0.1, 0.9, text1, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][2].text(0.1, 0.9, text2, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][4].text(0.1, 0.9, text3, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

    plt.subplots_adjust(wspace=0, hspace=0)
    pos1 = axes[0, 0].get_position()
    y_pos = pos1.y1 + 2*(pos1.y1 - pos1.y0)/100
    fig.text((pos1.x0 + pos1.x1) / 2, y_pos, 'Col1 $F_i$', ha='center', fontsize=16)
    pos2 = axes[0, 1].get_position()
    fig.text((pos2.x0 + pos2.x1) / 2, y_pos, 'Col2 $F_f$', ha='center', fontsize=16)
    pos3 = axes[0, 2].get_position()
    fig.text((pos3.x0 + pos3.x1) / 2, y_pos, 'Col3 $\hat F_f$', ha='center', fontsize=16)
    pos4 = axes[0, 3].get_position()
    fig.text((pos4.x0 + pos4.x1) / 2, y_pos, 'Col4 $F_b$', ha='center', fontsize=16)
    pos5 = axes[0, 4].get_position()
    fig.text((pos5.x0 + pos5.x1) / 2, y_pos, 'Col5 $\hat F_b$', ha='center', fontsize=16)
    pos6 = axes[0, 5].get_position()
    fig.text((pos6.x0 + pos6.x1) / 2, y_pos, 'Col6 $F_n$', ha='center', fontsize=16)
    pos7 = axes[0, 6].get_position()
    fig.text((pos7.x0 + pos7.x1) / 2, y_pos, 'Col7 $\hat F_n$', ha='center', fontsize=16)
    plt.savefig(f'figures/{save_name}_sample.pdf',  bbox_inches='tight', pad_inches=0.0)



#==== show the real image reconstruction
def display_real_image(data, size=2, save_name='pred_real'):

    input = data['input']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    RA, DEC = data['RA'], data['DEC']
    residual = input - pred_fore - pred_back
    chi1, chi2 = calculate_real_chi(data)
    fig, axes = plt.subplots(input.shape[0], 4, figsize=(4*size, size*input.shape[0]), sharex=False, sharey=False)
    for i in range(input.shape[0]):
        
        RGB_image = input[i][[3,2,1]]  # Conventionally R = i', G = r', and B = g'.
        fore_image= pred_fore[i][[3,2,1]]
        back_image= pred_back[i][[3,2,1]]
        resid = residual[i][[3,2,1]]
        text1 = 'ra=' + str(round(RA[i],3)) + '\ndec=' + str(round(DEC[i],3))
        text2 = '$\chi_1^2=$'+str(round(chi1[i],2))
        text3 = '$\chi_2^2=$'+str(round(chi2[i],2))

        rad = 10
        size = residual.shape[-1]
        norm1 = np.max(RGB_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(RGB_image.shape[0],1,1)
        norm2 = np.max(fore_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(fore_image.shape[0],1,1)
        norm3 = np.max(back_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(back_image.shape[0],1,1)
        norm4 = 10*np.median(np.abs(resid), (-2,-1)).reshape(resid.shape[0],1,1)
        axes[i][0].imshow(np.transpose(RGB_image/norm1, (1, 2, 0)))
        axes[i][0].text(0.1, 0.9, text1, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        
        axes[i][1].imshow(np.transpose(fore_image/norm1, (1, 2, 0)))
        axes[i][1].text(0.1, 0.9, text2, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

        axes[i][2].imshow(np.transpose(5*back_image/norm1, (1, 2, 0)))
        axes[i][2].text(0.1, 0.9, text3, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

        axes[i][3].imshow(np.transpose(resid/norm4, (1, 2, 0)))

        axes[i][0].xaxis.set_visible(False)
        axes[i][0].yaxis.set_visible(False)
        axes[i][1].xaxis.set_visible(False)
        axes[i][1].yaxis.set_visible(False)
        axes[i][2].xaxis.set_visible(False)
        axes[i][2].yaxis.set_visible(False)
        axes[i][3].xaxis.set_visible(False)
        axes[i][3].yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    pos1 = axes[0, 0].get_position()
    y_pos = pos1.y1 + 2*(pos1.y1 - pos1.y0)/100
    fig.text((pos1.x0 + pos1.x1) / 2, y_pos, 'Col1 $F_i$', ha='center', fontsize=16)
    pos2 = axes[0, 1].get_position()
    fig.text((pos2.x0 + pos2.x1) / 2, y_pos, 'Col2 $\hat F_f$', ha='center', fontsize=16)
    pos3 = axes[0, 2].get_position()
    fig.text((pos3.x0 + pos3.x1) / 2, y_pos, 'Col3 $\hat F_b$', ha='center', fontsize=16)
    pos4 = axes[0, 3].get_position()
    fig.text((pos4.x0 + pos4.x1) / 2, y_pos, 'Col4 $\hat F_n$', ha='center', fontsize=16)
    plt.savefig(f'figures/{save_name}_sample.pdf',  bbox_inches='tight', pad_inches=0.0)

#==== show the real image reconstruction
def display_real_HST_image(data, size=2, save_name='pred_HST_real'):

    input = data['input']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    id = data['id']
    residual = input - pred_fore - pred_back
    _, _, chi1, chi2 = calculate_HST_chi(data)
    fig, axes = plt.subplots(input.shape[0], 4, figsize=(4*size, size*input.shape[0]), sharex=False, sharey=False)
    for i in range(input.shape[0]):
        
        RGB_image = input[i]  # Conventionally R = i', G = r', and B = g'.
        fore_image= pred_fore[i]
        back_image= pred_back[i]
        resid = residual[i]
        text1 = id[i][3:]
        text2 = '$\chi_1^2=$'+str(round(chi1[i],2))
        text3 = '$\chi_2^2=$'+str(round(chi2[i],2))

        rad = 10
        size = residual.shape[-1]
        norm1 = np.max(RGB_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(RGB_image.shape[0],1,1)
        norm2 = np.max(fore_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(fore_image.shape[0],1,1)
        norm3 = np.max(back_image[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(back_image.shape[0],1,1)
        norm4 = 10*np.median(np.abs(resid), (-2,-1)).reshape(resid.shape[0],1,1)
        axes[i][0].imshow(np.transpose(RGB_image/norm1, (1, 2, 0)))
        axes[i][0].text(0.1, 0.9, text1, fontsize=8, horizontalalignment='left', verticalalignment='top', color='white')
        
        axes[i][1].imshow(np.transpose(fore_image/norm1, (1, 2, 0)))
        axes[i][1].text(0.1, 0.9, text2, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

        axes[i][2].imshow(np.transpose(5*back_image/norm1, (1, 2, 0)))
        axes[i][2].text(0.1, 0.9, text3, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

        axes[i][3].imshow(np.transpose(resid/norm4, (1, 2, 0)))

        axes[i][0].xaxis.set_visible(False)
        axes[i][0].yaxis.set_visible(False)
        axes[i][1].xaxis.set_visible(False)
        axes[i][1].yaxis.set_visible(False)
        axes[i][2].xaxis.set_visible(False)
        axes[i][2].yaxis.set_visible(False)
        axes[i][3].xaxis.set_visible(False)
        axes[i][3].yaxis.set_visible(False)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(f'figures/{save_name}_sample.pdf',  bbox_inches='tight', pad_inches=0.0)


def output_example_image(data, mask_output=False, size=2):

    input = data['input']
    fore_image= data['pred_fore']
    back_image= data['pred_back']
    pred_noise= data['pred_noise']
    if mask_output:
        residual = pred_noise
    else:
        residual = input - fore_image - back_image
    #if mask_output:
    #    residual = pred_noise.detach().numpy()

    rad = 25
    for i in range(input.shape[0]):     
        RGB_input = input[i][[3,2,1]]  # Conventionally R = i', G = r', and B = g'.
        RGB_fore= fore_image[i][[3,2,1]]
        RGB_back= back_image[i][[3,2,1]]
        RGB_residual = residual[i][[3,2,1]]
    
        size = RGB_residual.shape[-1]
        norm1 = np.max(RGB_input[:,size//2-rad:size//2+rad,size//2-rad:size//2+rad], (-2,-1)).reshape(RGB_input.shape[0],1,1)
        norm4 = np.max(RGB_residual, (-2,-1)).reshape(RGB_residual.shape[0],1,1)
        
        f, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
        ax.imshow(np.transpose(RGB_input/norm1, (1, 2, 0)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        text = '$input$'
        ax.text(0.1, 0.9, text, fontsize=24, horizontalalignment='left', verticalalignment='top', color='white')
        plt.savefig(f'figures/sample_{i}_{mask_output}_input.pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
        ax.imshow(np.transpose(RGB_fore/norm1, (1, 2, 0)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if mask_output:
            text = '$mask_f$'
        else:
            text = '$\hat F_f$'
        ax.text(0.1, 0.9, text, fontsize=24, horizontalalignment='left', verticalalignment='top', color='white')
        plt.savefig(f'figures/sample_{i}_{mask_output}_fore.pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
        ax.imshow(np.transpose(5*RGB_back/norm1, (1, 2, 0)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if mask_output:
            text = '$mask_b$'
        else:
            text = '$\hat F_b$'
        ax.text(0.1, 0.9, text, fontsize=24, horizontalalignment='left', verticalalignment='top', color='white')
        plt.savefig(f'figures/sample_{i}_{mask_output}_back.pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close()

        f, ax = plt.subplots(1, 1, figsize=(4, 4), sharex=False, sharey=False)
        ax.imshow(np.transpose(RGB_residual/norm4, (1, 2, 0)))
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        if mask_output:
            text = '$mask_n$'
            color = 'black'
        else:
            text = '$\hat F_n$'
            color = 'white'
        ax.text(0.1, 0.9, text, fontsize=24, horizontalalignment='left', verticalalignment='top', color=color)
        plt.savefig(f'figures/sample_{i}_{mask_output}_residual.pdf', bbox_inches='tight', pad_inches=0.0)
        plt.close() 

# only r-band here
def display_HST_image(data, size=2, save_name='real_HST_lens'):

    input = data['input']
    fore_image = data['fore_image']
    back_image = data['back_image']
    pred_fore = data['pred_fore']
    pred_back = data['pred_back']
    pred_noise = data['pred_noise']
    noise = data['noise']
    weight = data['weight']
    id = data['id']
    model_chi1, model_chi2, chi1, chi2 = calculate_HST_chi(data)

    image_list = [input, fore_image, pred_fore, back_image, pred_back, noise, pred_noise]
    fig, axes = plt.subplots(input.shape[0], len(image_list), figsize=(size*len(image_list), size*input.shape[0]), sharex=False, sharey=False)
    rad = 10
    for i in range(input.shape[0]):
        for j in range(len(image_list)):
            image = image_list[j][i]
            image = image[0] #only r-band here
            size = image.shape[-1]
            cal_norm = image_list[j-j%2][i][0] #
            if j < 5:
                norm = np.max(cal_norm[size//2-rad:size//2+rad,size//2-rad:size//2+rad])
            else:
                norm = np.median(np.abs(cal_norm))

            axes[i][j].imshow(image/norm)
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)
            axes[i][j].xaxis.set_visible(False)
            axes[i][j].yaxis.set_visible(False)

        text1 = id[i].replace('id=','')
        text2 = '$\chi_1^2=$'+str(round(model_chi1[i],2))
        text3 = '$\chi_1^2=$'+str(round(chi1[i],2))
        text4 = '$\chi_2^2=$'+str(round(model_chi2[i],2))
        text5 = '$\chi_2^2=$'+str(round(chi2[i],2))
        axes[i][0].text(0.1, 0.9, text1, fontsize=8, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][1].text(0.1, 0.9, text2, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][2].text(0.1, 0.9, text3, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][3].text(0.1, 0.9, text4, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')
        axes[i][4].text(0.1, 0.9, text5, fontsize=16, horizontalalignment='left', verticalalignment='top', color='white')

    plt.subplots_adjust(wspace=0, hspace=0)
    pos1 = axes[0, 0].get_position()
    y_pos = pos1.y1 + 2*(pos1.y1 - pos1.y0)/100
    fig.text((pos1.x0 + pos1.x1) / 2, y_pos, 'Col1 $F_i$', ha='center', fontsize=16)
    pos2 = axes[0, 1].get_position()
    fig.text((pos2.x0 + pos2.x1) / 2, y_pos, 'Col2 $F_f$', ha='center', fontsize=16)
    pos3 = axes[0, 2].get_position()
    fig.text((pos3.x0 + pos3.x1) / 2, y_pos, 'Col3 $\hat F_f$', ha='center', fontsize=16)
    pos4 = axes[0, 3].get_position()
    fig.text((pos4.x0 + pos4.x1) / 2, y_pos, 'Col4 $F_b$', ha='center', fontsize=16)
    pos5 = axes[0, 4].get_position()
    fig.text((pos5.x0 + pos5.x1) / 2, y_pos, 'Col5 $\hat F_b$', ha='center', fontsize=16)
    pos6 = axes[0, 5].get_position()
    fig.text((pos6.x0 + pos6.x1) / 2, y_pos, 'Col6 $F_n$', ha='center', fontsize=16)
    pos7 = axes[0, 6].get_position()
    fig.text((pos7.x0 + pos7.x1) / 2, y_pos, 'Col7 $\hat F_n$', ha='center', fontsize=16)
    plt.savefig(f'figures/{save_name}_sample.pdf',  bbox_inches='tight', pad_inches=0.0)
    


