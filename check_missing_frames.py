# -*- coding: utf-8 -*-
"""
检查 RGB 序列目录中缺失（空缺）的帧。
用法:
    python check_missing_frames.py                      # 默认 ./demo_data/tooth/rgb
    python check_missing_frames.py --dir /path/to/rgb
    python check_missing_frames.py --dir ./demo_data/tooth/rgb --ext png jpg

命名兼容: frame_0001.png / frame0001.png / frame-0001.png / 0001.png（自动提取数字帧号）。
输出: 总帧数、帧号范围、补零位数、连续段、缺失帧号列表。
"""
import os
import re
import argparse


def extract_frame_number(filename):
    # 取文件名中最后一段连续数字作为帧号（兼容 frame_0001 / frame0001 / frame-0001 / 0001）
    name, ext = os.path.splitext(filename)
    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
        return None
    m = re.findall(r"(\d+)", name)
    if not m:
        return None
    return int(m[-1])


def fmt(n, width):
    return str(n).zfill(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join("demo_data", "tooth", "rgb"),
                    help="帧目录，默认 demo_data/tooth/rgb")
    args = ap.parse_args()

    d = args.dir
    if not os.path.isdir(d):
        print(f"[错误] 目录不存在: {os.path.abspath(d)}")
        return

    frame2name = {}
    for fn in os.listdir(d):
        n = extract_frame_number(fn)
        if n is not None:
            frame2name[n] = fn

    if not frame2name:
        print(f"[提示] 在 {os.path.abspath(d)} 下未找到 frame_XXXX 形式的图片。")
        return

    frames = sorted(frame2name)
    fmin, fmax = frames[0], frames[-1]
    # 推断补零位数（取任一文件名里数字部分的长度）
    sample = re.findall(r"\d+", frame2name[frames[0]])[-1]
    width = len(sample)

    found = set(frames)
    missing = [n for n in range(fmin, fmax + 1) if n not in found]

    # 连续存在区间（用于快速看断点）
    runs, start, prev = [], frames[0], frames[0]
    for n in frames[1:]:
        if n == prev + 1:
            prev = n
        else:
            runs.append((start, prev))
            start = prev = n
    runs.append((start, prev))

    print(f"目录: {os.path.abspath(d)}")
    print(f"图片总数(可识别帧号): {len(frames)}")
    print(f"帧号范围: {fmt(fmin,width)} ~ {fmt(fmax,width)} (补零 {width} 位)")
    print(f"理论应连续帧数: {fmax - fmin + 1}；实际 {len(frames)}；"
          f"缺失 {len(missing)} 帧")
    print("-" * 60)

    if missing:
        # 把缺失帧也连成连续段显示
        mruns, ms, mp = [], missing[0], missing[0]
        for n in missing[1:]:
            if n == mp + 1:
                mp = n
            else:
                mruns.append((ms, mp)); ms = mp = n
        mruns.append((ms, mp))
        print("缺失帧号（连续段合并显示）:")
        for a, b in mruns:
            if a == b:
                print(f"  - 缺 {fmt(a,width)}")
            else:
                print(f"  - 缺 {fmt(a,width)} ~ {fmt(b,width)} （共 {b-a+1} 帧）")
        print("-" * 60)
        print("缺失帧完整列表:")
        print("  " + ", ".join(fmt(n, width) for n in missing))
    else:
        print("未发现缺失帧，序列连续。")

    print("-" * 60)
    print("存在帧的连续段:")
    for a, b in runs:
        tag = "" if a == b else f"  ({b-a+1} 帧)"
        print(f"  {fmt(a,width)} ~ {fmt(b,width)}{tag}")


if __name__ == "__main__":
    main()
