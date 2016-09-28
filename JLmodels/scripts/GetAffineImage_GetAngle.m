function ang = GetAffineImage_GetAngle(landmark)
left_eye_x = landmark(1);
left_eye_y = landmark(2);
right_eye_x = landmark(3);
right_eye_y = landmark(4);
ang = atan2((right_eye_y - left_eye_y),...
    (right_eye_x - left_eye_x));
end