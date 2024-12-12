import json
import requests
import logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT)


class DowndloadEngine(object):
    def download_img_from_url(self, url):
        """
        通过url下载图片
        :param url:
        :return:
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36 ",
            'Connection': 'close'
        }
        r = requests.get(url, headers=headers)
        r.close()
        return r.content

    def get_img_id_by_vec_id(self, vec_id_list):
        """
        获取图片id通过向量id
        :param vec_id_list:
        :return:
        """
        if len(vec_id_list) == 0:
            return []

        else:
            vec_id_list = ','.join([str(i) for i in vec_id_list])
            params = {'appid': '',
                      'sign': '',
                      'vec_id_list': vec_id_list
                      }
            r = requests.get(f'{self.api}/api/listImgIdByVecId', params=params)
            r.close()
            img_id_list = json.loads(r.content)['data']['list']
            logging.info(img_id_list)
            img_id_list = [i['img_id'] for i in img_id_list]
            return img_id_list


