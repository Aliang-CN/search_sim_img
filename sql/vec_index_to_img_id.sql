create table `vec_index_to_img_id` (
	`img_id` varchar( 100 ) comment '图片id',
	`vec_id` varchar( 100 ) comment '向量索引',
	`embedding` text comment '图片向量',
	`create_time` datetime DEFAULT current_timestamp comment '创建时间',
	`update_time` datetime DEFAULT current_timestamp on update CURRENT_TIMESTAMP comment '更新时间',
	primary key ( `img_id` ),
	unique index `img_id` ( `img_id` ),
  unique index `vec_id` ( `vec_id` ),
);