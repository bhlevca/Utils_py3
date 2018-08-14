
def ff_spec_rot(u, v):
    '''
    % ff_spec_rot.m -> compute the rotary spectra from u,v velocity components
    %
    % use:  [puv,quv,cw,ccw] = ff_spec_rot(u,v);
    % input:
    %        u-component of velocity
    %        v-component of velocity
    %
    % output:
    %        cw  - clockwise spectrum
    %        ccw - counter-clockwise spectrum
    %        puv - cross spectra
    %        quv - quadrature spectra
    %
    % example:
    %    [puv,quv,cw,ccw] = ff_spec_rot(u,v)
    %
    % other m-files required: ufft.m
    %
    % author:   Filipe P. A. Fernandes
    % e-mail:   ocefpaf@gmail.com
    % web:      http://ocefpaf.tiddlyspot.com/
    % date:     19-Jan-2002
    % modified: 06-May-2009 (translated to english)
    %
    % obs: based on J. Gonella Deep Sea Res., 833-846, 1972
    %      definition: The spectral energy at some frequency can be
    %      decomposed into two circulaly polarized constituents, one
    %      rotating clockwise and other anti-clockwise
    %
    % autospectra of the scalar components

        pu = fu.*conj(fu); pv = fv.*conj(fv);

        % cross spectra
        puv =  real(fu).*real(fv) + imag(fu).*imag(fv);

        % quadrature spectra
        quv = -real(fu).*imag(fv) + real(fv).*imag(fu);

        % rotatory components
        cw   = (pu+pv-2*quv)./8;
        ccw  = (pu+pv+2*quv)./8;
    '''

    # individual components fourier series
    fu = fft(u); fv = fft(v);

    # autospectra of the scalar components
    pu = fu * fu.conj()
    pv = fv * fv.conj()

    # cross spectra
    # puv =  real(fu).*real(fv) + imag(fu).*imag(fv);
    puv = fu.real * fv.real + fu.imag * fv.imag

    # quadrature spectra
    # quv = -real(fu).*imag(fv) + real(fv).*imag(fu);
    quv = -fu.real * fv.imag + fv.real * fu.imag

    # rotatory components
    cw = (pu + pv - 2 * quv) / 8
    ccw = (pu + pv + 2 * quv) / 8
    return [puv, quv, cw, ccw]
