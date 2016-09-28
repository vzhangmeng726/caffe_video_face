function scale = GetAffineImage_GetScale(parm, landmark)
left_eye_x = landmark(1);
left_eye_y = landmark(2);
right_eye_x = landmark(3);
right_eye_y = landmark(4);
left_mouth_x = landmark(7);
left_mouth_y = landmark(8);
right_mouth_x = landmark(9);
right_mouth_y = landmark(10);

norm_standard_len = parm.patchsize * parm.norm_ratio;

switch parm.mode
    case 'AVE_LE2LM_RE2RM'
        deltaX1 = left_eye_x - left_mouth_x;
        deltaY1 = left_eye_y - left_mouth_y;
        
        deltaX2 = right_eye_x - right_mouth_x;
        deltaY2 = right_eye_y - right_mouth_y;
        
        actual_len = sqrt(deltaX1 * deltaX1 + deltaY1 * deltaY1) + ...
            sqrt(deltaX2 * deltaX2 + deltaY2 * deltaY2);
        actual_len = actual_len / 2;
    case 'RECT_LE_RE_LM_RM'
        left_top_x = min(min(min(left_eye_x, right_eye_x),...
            left_mouth_x),...
            right_mouth_x);
        right_bottom_x = max(max(max(left_eye_x, right_eye_x),...
            left_mouth_x),...
            right_mouth_x);
        left_top_y = min(min(min(left_eye_y, right_eye_y),...
            left_mouth_y),...
            right_mouth_y);
        right_bottom_y = max(max(max(left_eye_y, right_eye_y),...
            left_mouth_y),...
            right_mouth_y);
        deltaX = right_bottom_x - left_top_x;
        deltaY = right_bottom_y - left_top_y;
        actual_len = sqrt((deltaX.^2 + deltaY.^2)/2);
    otherwise
         error('Unkonw Norm Mode.');
end

scale = actual_len / norm_standard_len;
end