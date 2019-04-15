from astropy.wcs import WCS




def tweak_header(header):
    '''
    Header tweaks to make the 3D kcwi header compatible with astropy image header object
    '''
    header['NAXIS']=2
    header['WCSDIM']=2
    if 'CRVAL3' in header.keys():
        header.remove('CRVAL3')
    if 'CRPIX3' in header.keys():
        header.remove('CRPIX3')
    if 'CD3_3' in header.keys():
        header.remove('CD3_3')
    if 'NAXIS3' in header.keys():
        header.remove('NAXIS3')
    if 'CUNIT3' in header.keys():
        header.remove('CUNIT3')
    if 'CTYPE3' in header.keys():
        header.remove('CTYPE3')
    if 'CNAME3' in header.keys():
        header.remove('CNAME3')

    return header
