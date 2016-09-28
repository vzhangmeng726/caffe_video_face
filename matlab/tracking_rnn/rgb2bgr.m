function bgr=rgb2bgr(rgb)

   b = rgb(:,:,3);
   rgb(:,:,3) = rgb(:,:,1);
   rgb(:,:,1) = b;
   bgr = rgb;

end