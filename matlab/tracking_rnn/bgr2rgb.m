function rgb=bgr2rgb(bgr)

   r = bgr(:,:,3);
   bgr(:,:,3) = bgr(:,:,1);
   bgr(:,:,1) = r;
   rgb = bgr;

end