create table `sim_img` (
	`img_id` varchar( 100 ) comment '图片id',
	`sim_img_list` text comment '相似图片',
	`create_time` datetime DEFAULT current_timestamp comment '创建时间',
	`update_time` datetime DEFAULT current_timestamp on update CURRENT_TIMESTAMP comment '更新时间',
	primary key ( `img_id` )
);