"""設定の読み込み。既定値は default_config.yaml だけに書き、ユーザー設定で上書きする。"""
import copy
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name('default_config.yaml')


class Config(dict):
    """cfg['jitter']['one_euro'] と cfg.jitter.one_euro の両方で読める dict。"""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self):
        return {k: (v.to_dict() if isinstance(v, Config) else copy.deepcopy(v))
                for k, v in self.items()}


def _wrap(d):
    if isinstance(d, dict):
        return Config({k: _wrap(v) for k, v in d.items()})
    return d


def _merge(base, update, path=''):
    # 既定値に無いキーはタイプミスとみなしてエラーにする
    for key, value in update.items():
        where = f'{path}.{key}' if path else key
        if key not in base:
            raise KeyError(f'未知の設定項目です: {where}（default_config.yaml を確認してください）')
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise TypeError(f'{where} は辞書で指定してください')
            _merge(base[key], value, where)
        else:
            base[key] = value


def _load_file(path):
    text = Path(path).read_text(encoding='utf-8')
    if str(path).lower().endswith('.json'):
        return json.loads(text) or {}
    return yaml.safe_load(text) or {}


def parse_override(item):
    """'center.mode=B' のような 1 行を {'center': {'mode': 'B'}} にする（値は YAML として解釈）。"""
    key, sep, value = item.partition('=')
    if not sep:
        raise ValueError(f'"キー=値" の形で指定してください: {item}')
    node = out = {}
    parts = key.strip().split('.')
    for p in parts[:-1]:
        node[p] = {}
        node = node[p]
    node[parts[-1]] = yaml.safe_load(value)
    return out


def load_config(config=None, overrides=None):
    """既定値に config（パス / dict / Config / None）と overrides（'a.b=c' のリスト）を重ねる。"""
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding='utf-8'))
    if config is not None:
        user = _load_file(config) if isinstance(config, (str, Path)) else config
        _merge(cfg, dict(user))
    for item in overrides or []:
        _merge(cfg, parse_override(item) if isinstance(item, str) else item)
    return _wrap(cfg)


def dump_config(cfg, path):
    data = cfg.to_dict() if isinstance(cfg, Config) else cfg
    Path(path).write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
                          encoding='utf-8')
