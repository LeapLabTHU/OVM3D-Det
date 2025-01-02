import torch
import numpy as np
import torch.nn.functional as F

def calc_dis_ray_tracing(wl, Ry, points, bev_box_center):
    init_theta, length = torch.atan(wl[0] / wl[1]), torch.sqrt(wl[0] ** 2 + wl[1] ** 2) / 2  # 0.5:1
    corners = [((length * torch.cos(init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi - init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi - init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(np.pi + init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(np.pi + init_theta + Ry) + bev_box_center[1]).unsqueeze(0)),

               ((length * torch.cos(-init_theta + Ry) + bev_box_center[0]).unsqueeze(0),
                (length * torch.sin(-init_theta + Ry) + bev_box_center[1]).unsqueeze(0))]
    if Ry == np.pi/2:
        Ry -= 1e-4
    if Ry == 0:
        Ry += 1e-4
    k1, k2 = torch.tan(Ry), torch.tan(Ry + np.pi / 2)
    b11 = corners[0][1] - k1 * corners[0][0]
    b12 = corners[2][1] - k1 * corners[2][0]
    b21 = corners[0][1] - k2 * corners[0][0]
    b22 = corners[2][1] - k2 * corners[2][0]

    line0 = [k1, -1, b11]
    line1 = [k2, -1, b22]
    line2 = [k1, -1, b12]
    line3 = [k2, -1, b21]

    points[points[:, 0] == 0, 0] = 1e-4 # avoid inf

    slope_x = points[:, 1] / points[:, 0]
    intersect0 = torch.stack([line0[2] / (slope_x - line0[0]),
                              line0[2]*slope_x / (slope_x - line0[0])], dim=1)
    intersect1 = torch.stack([line1[2] / (slope_x - line1[0]),
                              line1[2]*slope_x / (slope_x - line1[0])], dim=1)
    intersect2 = torch.stack([line2[2] / (slope_x - line2[0]),
                              line2[2]*slope_x / (slope_x - line2[0])], dim=1)
    intersect3 = torch.stack([line3[2] / (slope_x - line3[0]),
                              line3[2]*slope_x / (slope_x - line3[0])], dim=1)


    dis0 = torch.abs(intersect0[:, 0] - points[:, 0]) + torch.abs(intersect0[:, 1] - points[:, 1])
    dis1 = torch.abs(intersect1[:, 0] - points[:, 0]) + torch.abs(intersect1[:, 1] - points[:, 1])
    dis2 = torch.abs(intersect2[:, 0] - points[:, 0]) + torch.abs(intersect2[:, 1] - points[:, 1])
    dis3 = torch.abs(intersect3[:, 0] - points[:, 0]) + torch.abs(intersect3[:, 1] - points[:, 1])


    dis_inter2center0 = torch.sqrt(intersect0[:, 0]**2 + intersect0[:, 1]**2)
    dis_inter2center1 = torch.sqrt(intersect1[:, 0]**2 + intersect1[:, 1]**2)
    dis_inter2center2 = torch.sqrt(intersect2[:, 0]**2 + intersect2[:, 1]**2)
    dis_inter2center3 = torch.sqrt(intersect3[:, 0]**2 + intersect3[:, 1]**2)

    intersect0 = torch.round(intersect0*1e4)
    intersect1 = torch.round(intersect1*1e4)
    intersect2 = torch.round(intersect2*1e4)
    intersect3 = torch.round(intersect3*1e4)

    dis0_in_box_edge = ((intersect0[:, 0] > torch.round(min(corners[0][0], corners[1][0])*1e4)) &
                        (intersect0[:, 0] < torch.round(max(corners[0][0], corners[1][0])*1e4))) | \
                       ((intersect0[:, 1] > torch.round(min(corners[0][1], corners[1][1])*1e4)) &
                        (intersect0[:, 1] < torch.round(max(corners[0][1], corners[1][1])*1e4)))
    dis1_in_box_edge = ((intersect1[:, 0] > torch.round(min(corners[1][0], corners[2][0])*1e4)) &
                        (intersect1[:, 0] < torch.round(max(corners[1][0], corners[2][0])*1e4))) | \
                       ((intersect1[:, 1] > torch.round(min(corners[1][1], corners[2][1])*1e4)) &
                        (intersect1[:, 1] < torch.round(max(corners[1][1], corners[2][1])*1e4)))
    dis2_in_box_edge = ((intersect2[:, 0] > torch.round(min(corners[2][0], corners[3][0])*1e4)) &
                        (intersect2[:, 0] < torch.round(max(corners[2][0], corners[3][0])*1e4))) | \
                       ((intersect2[:, 1] > torch.round(min(corners[2][1], corners[3][1])*1e4)) &
                        (intersect2[:, 1] < torch.round(max(corners[2][1], corners[3][1])*1e4)))
    dis3_in_box_edge = ((intersect3[:, 0] > torch.round(min(corners[3][0], corners[0][0])*1e4)) &
                        (intersect3[:, 0] < torch.round(max(corners[3][0], corners[0][0])*1e4))) | \
                       ((intersect3[:, 1] > torch.round(min(corners[3][1], corners[0][1])*1e4)) &
                        (intersect3[:, 1] < torch.round(max(corners[3][1], corners[0][1])*1e4)))

    dis_in_mul = torch.stack([dis0_in_box_edge, dis1_in_box_edge,
                              dis2_in_box_edge, dis3_in_box_edge], dim=1)
    dis_inter2cen = torch.stack([dis_inter2center0, dis_inter2center1,
                                 dis_inter2center2, dis_inter2center3], dim=1)
    dis_all = torch.stack([dis0, dis1, dis2, dis3], dim=1)

    dis_in = (torch.sum(dis_in_mul, dim=1) == 2).type(torch.bool)
    if torch.sum(dis_in.int()) < 3:
        return 0

    dis_in = dis_in.squeeze(0)
    dis_in_mul = dis_in_mul[:,:,dis_in]
    dis_inter2cen = dis_inter2cen[:,:,dis_in]
    dis_all = dis_all[:,:,dis_in]

    z_buffer_ind = torch.argmin(dis_inter2cen[dis_in_mul].view(2, -1), dim=0)
    z_buffer_ind_gather = torch.stack([~z_buffer_ind.byte(), z_buffer_ind.byte()],
                                      dim=1).type(torch.bool)
    dis = (dis_all[dis_in_mul].view(2, -1).permute(1,0))[z_buffer_ind_gather]

    dis_mean = torch.mean(dis)

    return dis_mean.item()



def calc_inside_ratio(pc, x_min, x_max, z_min, z_max):
    inside = (pc[0, :] > x_min) & (pc[0, :] < x_max) & \
             (pc[2, :] > z_min) & (pc[2, :] < z_max)
    inside_ratio = np.sum(inside) / pc.shape[1]
    return inside_ratio