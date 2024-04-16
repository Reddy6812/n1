% I have verified that this exe based PESQ 
% produce the same scores as the codes
% in the outer folder
% source: https://www.mathworks.com/matlabcentral/fileexchange/47333-pesq-matlab-driver


% TEST_PESQ2_MTLB demonstrates use of the PESQ2_MTLB function
%
%   See also PESQ2_MTLB
%
%   Author: Arkadiy Prodeus, email: aprodeus@gmail.com, July 2014
close all; 
    % name of executable file for PESQ calculation
    binary = 'pesq2.exe';
    % specify path to folder with reference and degraded audio files in it
    pathaudio = '../';
    % specify reference and degraded audio files
    reference = 'arabic_source.wav';
    degraded = 'arabic_400mhz.wav';

    
    % compute NB-PESQ and WB-PESQ scores for wav-files
    nb = pesq2_mtlb( reference, degraded, 16000, 'nb', binary, pathaudio );
    wb = pesq2_mtlb( reference, degraded, 16000, 'wb', binary, pathaudio );
    
 
    % display results to screen
    fprintf('====================\n'); 
    disp('Example 1: compute NB-PESQ scores for wav-files:');
    fprintf( 'NB PESQ MOS = %5.3f\n', nb(1) );
    fprintf( 'NB MOS LQO  = %5.3f\n', nb(2) );
    


